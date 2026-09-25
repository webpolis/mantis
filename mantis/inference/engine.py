"""
MANTIS Inference Engine

Orchestrates dynamic routing and generation with memory systems.
"""

import contextlib
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mantis.configs.model_config import InferenceConfig
from mantis.inference.generation import generate_tokens
from mantis.inference.prompting import build_prompt_ids, evidence_ids, format_query, render_entry, select_evidence
from mantis.models.meta_controller import GATES, MetaController, StateSummaryEncoder

ABSTENTION = "I can't verify an answer to that from the available evidence, so I won't guess."
GLOBAL_NAMESPACE = 'global'


def build_policy(config) -> Tuple[MetaController, StateSummaryEncoder]:
    """Meta-controller and state encoder sized from a MANTISConfig."""
    mc = config.meta_controller
    meta = MetaController(
        d_model=config.base_moe.d_model,
        n_layers=mc.n_layers,
        d_ff=mc.d_ff,
        dropout=mc.dropout,
        n_experts=config.base_moe.n_experts,
        n_moe_layers=config.base_moe.n_layers,
        state_dim=mc.state_dim,
        expert_bias_scale=mc.expert_bias_scale,
    )
    return meta, StateSummaryEncoder(state_dim=mc.state_dim)


class MANTISInferenceEngine:
    """
    Main inference engine for MANTIS.

    Coordinates:
    - Meta-controller routing decisions (or a fixed policy for ablations)
    - Evidence retrieval from episodic and semantic memory under a token budget
    - Base model generation, reusing the query prefill whenever the prompt is unchanged
    - Critic verification with a bounded retrieve-regenerate-recheck recovery
    - Provenance-tagged episodic writes and the consolidation lifecycle

    Missing components (episodic, semantic, critic) disable their gate: the
    gate always reads 0 and does not enter policy log-probabilities.
    """

    def __init__(
        self,
        base_model,
        meta_controller,
        state_encoder,
        tokenizer,
        episodic_memory=None,
        semantic_memory=None,
        critic_model=None,
        config: Optional[InferenceConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        consolidator=None,
        memory_dir: Optional[str] = None,
    ):
        if tokenizer is None:
            raise ValueError("MANTISInferenceEngine requires a tokenizer")
        self.device = device
        self.base = base_model.to(device).eval()
        self.meta = meta_controller.to(device).eval()
        self.state_encoder = state_encoder.to(device).eval()
        self.critic = critic_model.to(device).eval() if critic_model is not None else None
        self.episodic = episodic_memory
        if episodic_memory is not None:
            episodic_memory.ssm.eval()
        self.semantic = semantic_memory
        self.consolidator = consolidator
        self.memory_dir = memory_dir
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        self.gate_mask = torch.tensor(
            [[1.0, float(episodic_memory is not None), float(semantic_memory is not None),
              float(critic_model is not None)]],
            device=device,
        )
        self.expert_enabled = self.config.expert_bias and self.base.n_experts > 1
        self.banned_ids = tokenizer.non_generable_ids
        self.active_params = self.base.count_parameters()['active']
        self.critic_params = sum(p.numel() for p in self.critic.parameters()) if self.critic else 0
        self.ssm_params = sum(p.numel() for p in self.episodic.ssm.parameters()) if self.episodic else 0

        self.reset_stats()

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint: Optional[str] = None,
        policy_checkpoint: Optional[str] = None,
        memory_checkpoint: Optional[str] = None,
        semantic_store: Optional[str] = None,
        critic_checkpoint: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        memory_dir: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> 'MANTISInferenceEngine':
        """
        Build the full engine from saved artifacts.

        A Stage 3 policy checkpoint records the paths of the components it was
        trained with; explicit arguments override them. `dtype` sets the
        backbone serving precision. `memory_dir` holds runtime memory state:
        stores found there are loaded (over the Stage 2 store), and the
        consolidator checkpoints both stores there. When episodic and semantic
        memory both exist, the consolidation lifecycle starts automatically;
        call close() to flush it.
        """
        from mantis.utils.checkpoints import check_tokenizer, compat_load, load_base_model, model_fingerprint

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        policy = compat_load(policy_checkpoint) if policy_checkpoint else {}
        base_checkpoint = base_checkpoint or policy.get('base_checkpoint')
        memory_checkpoint = memory_checkpoint or policy.get('memory_checkpoint')
        semantic_store = semantic_store or policy.get('semantic_store')
        critic_checkpoint = critic_checkpoint or policy.get('critic_checkpoint')
        if not base_checkpoint:
            raise ValueError("A base model checkpoint is required")

        base, tokenizer, base_ckpt = load_base_model(base_checkpoint, device, tokenizer_path, dtype)
        mantis_config = base_ckpt['config']
        fingerprint = getattr(base, '_source_fingerprint', None) or model_fingerprint(base, tokenizer)

        meta, state_encoder = build_policy(mantis_config)
        if policy:
            check_tokenizer(policy, tokenizer, policy_checkpoint)
            expected = policy.get('embedding_fingerprint')
            if expected is not None and expected != fingerprint:
                raise ValueError(f"Policy {policy_checkpoint} was trained with a different base model")
            meta.load_state_dict(policy['meta_controller_state_dict'])
            state_encoder.load_state_dict(policy['state_encoder_state_dict'])
        else:
            print("⚠️  No policy checkpoint: meta-controller is untrained")

        episodic = None
        if memory_checkpoint:
            mem = compat_load(memory_checkpoint)
            check_tokenizer(mem, tokenizer, memory_checkpoint)
            expected = mem.get('embedding_fingerprint')
            if expected is not None and expected != fingerprint:
                raise ValueError(f"Episodic checkpoint {memory_checkpoint} was trained with a different base model")
            from mantis.memory.episodic import EpisodicMemory
            from mantis.models.ssm import EpisodicMemorySSM
            em = mantis_config.episodic_memory
            ssm = EpisodicMemorySSM(d_model=mantis_config.base_moe.d_model, d_state=em.d_state, n_blocks=em.n_blocks,
                                    d_conv=em.d_conv, expand=em.expand, max_seq_len=em.max_seq_len,
                                    dropout=em.dropout)
            ssm.load_state_dict(mem['episodic_ssm_state_dict'])
            episodic = EpisodicMemory(ssm, max_entries=em.max_entries, context_window=em.max_seq_len, device=device)
            saved = memory_dir and os.path.join(memory_dir, 'episodic.pt')
            if saved and os.path.exists(saved):
                print(f"Loaded {episodic.load(saved)} episodic entries from {saved}")

        semantic = None
        saved_store = memory_dir and os.path.join(memory_dir, 'semantic')
        if saved_store and os.path.exists(f"{saved_store}.meta"):
            semantic_store = saved_store
        if semantic_store:
            from mantis.memory.semantic import SemanticMemory
            semantic = SemanticMemory.load(semantic_store, use_gpu=mantis_config.semantic_memory.use_gpu)
            if semantic.embedding_fingerprint and semantic.embedding_fingerprint != fingerprint:
                raise ValueError(
                    f"Semantic store {semantic_store} was built with a different base model or tokenizer "
                    f"(fingerprint {semantic.embedding_fingerprint}, model {fingerprint})"
                )

        critic = None
        if critic_checkpoint:
            from mantis.models.critic import CriticModel
            crit = compat_load(critic_checkpoint)
            check_tokenizer(crit, tokenizer, critic_checkpoint)
            expected = crit.get('embedding_fingerprint')
            if expected is not None and expected != fingerprint:
                raise ValueError(f"Critic checkpoint {critic_checkpoint} was trained with a different base model")
            critic = CriticModel.from_config(crit['config'].critic, mantis_config.base_moe)
            critic.load_state_dict(crit['critic_state_dict'])

        consolidator = None
        if episodic is not None and semantic is not None:
            from mantis.memory.consolidation import MemoryConsolidator
            em = mantis_config.episodic_memory
            consolidator = MemoryConsolidator(
                episodic, semantic, tokenizer,
                consolidation_interval=em.consolidation_interval,
                min_hits=em.consolidation_min_hits,
                queue_size=em.consolidation_queue_size,
                persist_dir=memory_dir,
            )
            consolidator.start()

        engine = cls(base, meta, state_encoder, tokenizer, episodic, semantic, critic,
                     config=config or mantis_config.inference, device=device,
                     consolidator=consolidator, memory_dir=memory_dir)
        engine.mantis_config = mantis_config
        return engine

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Flush consolidation and checkpoint memory state (if memory_dir is set)."""
        if self.consolidator is not None:
            self.consolidator.stop()
        if self.memory_dir:
            self.save_memory(self.memory_dir)

    def save_memory(self, directory: str) -> None:
        """Write the episodic buffer and semantic store to `directory`."""
        os.makedirs(directory, exist_ok=True)
        if self.episodic is not None:
            self.episodic.save(os.path.join(directory, 'episodic.pt'))
        if self.semantic is not None:
            self.semantic.save(os.path.join(directory, 'semantic'))

    @contextlib.contextmanager
    def frozen_memory(self):
        """
        Evaluate on a fixed memory: no interaction writes, no periodic
        consolidation. Independent-example benchmarks and RL validation
        must run inside this, or results depend on evaluation order.
        """
        remember = self.config.remember
        self.config.remember = False
        if self.consolidator is not None:
            self.consolidator.pause()
        try:
            yield self
        finally:
            self.config.remember = remember
            if self.consolidator is not None:
                self.consolidator.resume()

    def memory_stats(self) -> Dict:
        stats = {}
        if self.episodic is not None:
            stats['episodic'] = {'entries': self.episodic.size(), **self.episodic.stats}
        if self.semantic is not None:
            stats['semantic'] = {'entries': self.semantic.size(), **self.semantic.stats}
        if self.consolidator is not None:
            stats['consolidation'] = self.consolidator.get_stats()
        return stats

    # --------------------------------------------------------------- ingest

    @torch.no_grad()
    def ingest(self, text: str, namespace: str = 'default', source: str = 'document',
               metadata: Optional[Dict] = None) -> int:
        """
        Store a document in memory, chunked to the attention window.

        Chunks go to episodic memory (and reach semantic memory through
        consolidation), or straight to semantic memory when there is no
        episodic buffer. Returns the number of chunks written.
        """
        if self.episodic is None and self.semantic is None:
            raise ValueError("ingest() needs episodic or semantic memory")
        ids = self.tokenizer.encode(text)
        window = self.base.max_seq_len
        meta = {**(metadata or {}), 'namespace': namespace, 'source': source}
        chunks = [ids[i:i + window] for i in range(0, len(ids), window)]
        for chunk in chunks:
            hidden = self.base(torch.tensor([chunk], device=self.device), return_hidden=True)['last_hidden'][0].float()
            if self.episodic is not None:
                self.episodic.add(hidden, [('document', chunk)], meta)
            else:
                self.semantic.add(hidden.mean(dim=0).cpu(), self.tokenizer.decode(chunk), meta)
        return len(chunks)

    # ----------------------------------------------------------------- policy

    def _state_features(self, logits: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, float]:
        """
        Features for the state encoder, as one (1, 5) row:
        [query entropy, context fill, episodic fill, semantic non-empty, query top-1 probability].

        These describe how predictable the query text is, not whether the
        answer will be correct.
        """
        probs = torch.softmax(logits[0].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        query_entropy = float((entropy / torch.log(torch.tensor(float(probs.size(-1))))).mean())
        top1 = float(probs.max(dim=-1).values.mean())
        episodic_fill = self.episodic.size() / self.episodic.max_entries if self.episodic else 0.0
        semantic_ready = float(bool(self.semantic and self.semantic.size() > 0))
        features = torch.tensor(
            [[query_entropy, seq_len / self.base.max_seq_len, episodic_fill, semantic_ready, top1]],
            device=self.device,
        )
        return features, query_entropy

    def encode_state(self, features: torch.Tensor) -> torch.Tensor:
        """(batch, 5) features -> (batch, state_dim) summary."""
        return self.state_encoder(features[:, 0:1], features[:, 1:2], features[:, 2:4], features[:, 4:5])

    def _route(self, gate_actions: torch.Tensor) -> torch.Tensor:
        """Apply a fixed route policy (ablations) over the learned actions."""
        policy = self.config.route_policy
        if policy == 'learned':
            return gate_actions
        forced = torch.zeros_like(gate_actions)
        if policy == 'always':
            forced = self.gate_mask.clone()
        elif policy == 'bypass':
            forced[:, 0] = 1.0
        return forced

    # ------------------------------------------------------------- generation

    def _decode(self, prompt_ids: List[int], max_length: int, temperature: float, top_p: float,
                expert_weights: Optional[torch.Tensor], prefill, hidden_rows: Optional[List[torch.Tensor]],
                cost: Dict) -> Tuple[List[int], List[float]]:
        """
        Generate new tokens (EOS excluded) and their log-probabilities.

        `hidden_rows` (None when no memory can use them) collects the final
        hidden states of prompt + response for the memory write.
        """
        tokens, log_probs = [], []
        if prefill is None:
            cost['backbone_tokens'] += min(len(prompt_ids), self.base.max_seq_len)
        stopped = False
        for token, log_prob in generate_tokens(
            self.base, prompt_ids, max_length, temperature=temperature, top_p=top_p,
            banned_ids=self.banned_ids, stop_ids=[self.tokenizer.eos_token_id],
            expert_weights=expert_weights, prefill=prefill, hidden_out=hidden_rows,
        ):
            cost['backbone_tokens'] += 1
            if token == self.tokenizer.eos_token_id:
                stopped = True
                break
            tokens.append(token)
            log_probs.append(log_prob)
        if hidden_rows is not None and tokens and not stopped:
            cost['backbone_tokens'] += 1  # the last token's hidden state costs one more step
        return tokens, log_probs

    def _budget(self, query_len: int, max_length: int) -> int:
        """Evidence tokens that leave room for the query and half a window of generation."""
        reserve = min(max_length, self.base.max_seq_len // 2)
        return max(0, self.base.max_seq_len - query_len - reserve)

    def _retrieve(self, query_hidden: torch.Tensor, query_emb: torch.Tensor, gates: Dict[str, bool],
                  namespace: str, used: Dict[str, set], scale: int, cost: Dict) -> List[Dict]:
        """Candidate evidence from every open tier, with source identifiers and provenance."""
        cfg, items = self.config, []
        if gates['episodic']:
            cost['ssm_tokens'] += query_hidden.size(0)
            for entry in self.episodic.retrieve(query_hidden, cfg.episodic_top_k * scale, namespace, used['episodic']):
                meta = entry['metadata']
                items.append({
                    'id': f"E{entry['id']}", 'tier': 'episodic', 'origin': entry['id'],
                    'text': render_entry(entry, self.tokenizer), 'dense': (entry['score'] + 1) / 2,
                    'source': meta.get('source'), 'verified': meta.get('verified', False), 'trust': meta.get('trust', 0),
                })
        if gates['semantic']:
            hits = self.semantic.retrieve_with_metadata(
                query_emb, cfg.semantic_top_k * scale, namespaces={namespace, GLOBAL_NAMESPACE},
                min_trust=cfg.min_evidence_trust, exclude_ids=used['semantic'],
            )
            for hit in hits:
                meta = hit['metadata']
                items.append({
                    'id': f"S{hit['id']}", 'tier': 'semantic', 'origin': hit['id'], 'text': hit['text'],
                    'dense': min(1.0, max(0.0, 1 - hit['distance'] / 4)),
                    'source': meta.get('source'), 'verified': meta.get('verified', False), 'trust': meta.get('trust', 0),
                })
        return items

    def _verify(self, query_ids: List[int], response_ids: List[int], items: List[Dict], cost: Dict) -> float:
        input_ids, segment_ids = self.critic.build_input(query_ids, response_ids, evidence_ids(items))
        cost['backbone_tokens'] += len(input_ids)
        cost['critic_tokens'] += len(input_ids)
        return self.critic.score(self.base, input_ids, segment_ids)

    @torch.no_grad()
    def generate(
        self,
        query: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        namespace: str = 'default',
        evidence: Optional[Sequence[str]] = None,
        force_gates: Optional[Dict[str, bool]] = None,
        explore: bool = False,
        return_details: bool = False
    ) -> Dict:
        """
        Generate a response with dynamic routing.

        Args:
            query: Input query string. Input beyond the attention window is
                ingested into memory as document chunks before answering.
            max_length: Maximum new tokens (default: config)
            temperature: Sampling temperature (default: config)
            top_p: Nucleus sampling parameter (default: config)
            namespace: Memory namespace of this caller (plus shared 'global' facts)
            evidence: Oracle evidence texts used instead of retrieval (diagnostics)
            force_gates: Fixed gate decisions by name (route search and ablations);
                a forced bypass ignores the entropy threshold
            explore: Sample routing actions from the policy (RL rollouts)
            return_details: Include routing probabilities and the rollout record

        Returns:
            Dict with the response (completion only) and metadata: path,
            confidence and its source, critic score, abstention, evidence
            used, per-component timings and compute cost.
        """
        cfg = self.config
        max_length = cfg.max_length if max_length is None else max_length
        temperature = cfg.temperature if temperature is None else temperature
        top_p = cfg.top_p if top_p is None else top_p
        window = self.base.max_seq_len
        start_time = time.time()
        timings: Dict[str, float] = {}
        cost = {'backbone_tokens': 0, 'critic_tokens': 0, 'ssm_tokens': 0, 'retrievals': 0,
                'query_tokens': 0, 'generated_tokens': 0}

        prompt_ids = self.tokenizer.encode(format_query(query, cfg.prompt_format))
        if not prompt_ids:
            raise ValueError("Query is empty")
        if len(prompt_ids) > window and cfg.remember and (self.episodic is not None or self.semantic is not None):
            t = time.time()
            self.ingest(self.tokenizer.decode(prompt_ids[:-window]), namespace, source='user')
            timings['ingest'] = time.time() - t
        query_ids = prompt_ids[-window:]
        cost['query_tokens'] = len(query_ids)

        # Encode the query once; the cache and hidden states are reused below
        t = time.time()
        output = self.base(torch.tensor([query_ids], device=self.device), use_cache=True, return_hidden=True)
        cost['backbone_tokens'] += len(query_ids)
        hidden = output['last_hidden'][0].float()
        query_emb = hidden.mean(dim=0, keepdim=True)
        query_prefill = (output['past_key_values'], output['logits'][0, -1])
        timings['encode'] = time.time() - t

        t = time.time()
        features, query_entropy = self._state_features(output['logits'], len(query_ids))
        decisions = self.meta(query_emb, self.encode_state(features))
        gate_actions, expert_raw = self.meta.act(decisions, self.gate_mask, sample=explore, threshold=cfg.routing_threshold)
        gate_actions = self._route(gate_actions)
        if force_gates is not None:
            forced = torch.tensor([[float(force_gates.get(gate, False)) for gate in GATES]], device=self.device)
            gate_actions = forced * self.gate_mask
        gates = {gate: bool(v) for gate, v in zip(GATES, gate_actions[0].tolist())}
        forced_bypass = cfg.route_policy == 'bypass' or bool(force_gates and force_gates.get('bypass'))
        bypass = gates['bypass'] and (query_entropy < cfg.bypass_uncertainty_threshold or forced_bypass)

        action_mask = torch.cat([self.gate_mask, torch.tensor([[float(self.expert_enabled)]], device=self.device)], dim=1)
        if bypass:
            action_mask = action_mask.clone()
            action_mask[:, 1:] = 0.0
        log_prob = self.meta.log_prob(decisions, gate_actions, expert_raw, action_mask)
        expert_weights = self.meta.expert_bias(expert_raw) if self.expert_enabled and not bypass else None
        timings['route'] = time.time() - t

        used = {'episodic': set(), 'semantic': set()}
        items: List[Dict] = []
        oracle = evidence is not None
        if bypass:
            path = 'bypass'
            timings['retrieve'] = 0.0
        else:
            path = 'full'
            t = time.time()
            if oracle:
                candidates = [{'id': f"O{i}", 'tier': 'oracle', 'origin': i, 'text': text, 'dense': 1.0,
                               'source': 'external', 'verified': True, 'trust': 2} for i, text in enumerate(evidence)]
            else:
                candidates = self._retrieve(hidden, query_emb, gates, namespace, used, 1, cost)
                cost['retrievals'] += int(gates['episodic'] or gates['semantic'])
            items = select_evidence(candidates, query, self._budget(len(query_ids), max_length), self.tokenizer)
            timings['retrieve'] = time.time() - t

        t = time.time()
        want_hidden = (self.episodic is not None and cfg.remember) or (
            path == 'full' and gates['verification'] and cfg.verification_retries > 0
            and not oracle and (gates['episodic'] or gates['semantic'])
        )
        if items or expert_weights is not None:
            prefill, hidden_rows = None, ([] if want_hidden else None)
        else:
            prefill, hidden_rows = query_prefill, (list(hidden) if want_hidden else None)
        response_ids, log_probs = self._decode(
            build_prompt_ids(items, query_ids, self.tokenizer, cfg.prompt_format),
            max_length, temperature, top_p, expert_weights, prefill, hidden_rows, cost,
        )
        timings['generate'] = time.time() - t

        # Verification, with bounded evidence recovery before abstaining
        critic_score, abstained, rounds = None, False, 0
        timings['verify'] = 0.0
        if path == 'full' and gates['verification']:
            t = time.time()
            critic_score = self._verify(query_ids, response_ids, items, cost)
            can_retrieve = not oracle and (gates['episodic'] or gates['semantic'])
            while critic_score < cfg.verification_confidence_threshold and rounds < cfg.verification_retries and can_retrieve:
                rounds += 1
                for item in items:
                    used[item['tier']].add(item['origin'])
                probe = torch.stack(hidden_rows[-(len(query_ids) + len(response_ids)):]).float()
                more = self._retrieve(probe, probe.mean(dim=0, keepdim=True), gates, namespace, used, 2, cost)
                cost['retrievals'] += 1
                if not more:
                    break
                items = select_evidence(items + more, query, self._budget(len(query_ids), max_length), self.tokenizer)
                hidden_rows = []
                response_ids, log_probs = self._decode(
                    build_prompt_ids(items, query_ids, self.tokenizer, cfg.prompt_format),
                    max_length, temperature, top_p, expert_weights, None, hidden_rows, cost,
                )
                critic_score = self._verify(query_ids, response_ids, items, cost)
            abstained = critic_score < cfg.verification_confidence_threshold
            timings['verify'] = time.time() - t

        response = ABSTENTION if abstained else self.tokenizer.decode(response_ids)
        cost['generated_tokens'] = len(response_ids)

        # Remember what was actually asserted: never a rejected draft, never an abstention
        timings['remember'] = 0.0
        if self.episodic is not None and cfg.remember and not abstained:
            t = time.time()
            rows = hidden_rows[-(len(query_ids) + len(response_ids)):]
            self.episodic.add(torch.stack(rows), [('query', query_ids), ('response', response_ids)], {
                'namespace': namespace, 'source': 'interaction', 'verified': critic_score is not None,
                'trust': 1 if critic_score is not None else 0, 'critic_score': critic_score,
            })
            cost['ssm_tokens'] += len(rows)
            timings['remember'] = time.time() - t

        token_likelihood = float(torch.tensor(log_probs).mean().exp()) if log_probs else 0.0
        if abstained:
            confidence, confidence_source = 0.0, 'abstained'
        elif critic_score is not None:
            confidence, confidence_source = critic_score, 'critic'
        else:
            confidence, confidence_source = token_likelihood, 'token_likelihood'

        compute_units = (self.active_params * cost['backbone_tokens'] + self.critic_params * cost['critic_tokens']
                         + self.ssm_params * cost['ssm_tokens'])
        result = {
            'response': response,
            'path': path,
            'query_entropy': query_entropy,
            'token_likelihood': token_likelihood,
            'confidence': confidence,
            'confidence_source': confidence_source,
            'critic_score': critic_score,
            'abstained': abstained,
            'verified': critic_score is not None,
            'verification_rounds': rounds,
            'memory_used': {tier: any(item['tier'] == tier for item in items) for tier in ('episodic', 'semantic')},
            'evidence': [{k: item[k] for k in ('id', 'tier', 'source', 'score')} for item in items],
            'expert_bias_applied': expert_weights is not None,
            'num_tokens': 0 if abstained else len(response_ids),
            'latency': time.time() - start_time,
            'timings': timings,
            'cost': cost,
            'compute_units': compute_units,
        }

        if return_details:
            result['routing_decisions'] = gates
            result['routing_probs'] = {gate: float(decisions[gate]) for gate in GATES}
            result['rollout'] = {
                'query_emb': query_emb[0].cpu(),
                'features': features[0].cpu(),
                'gate_actions': gate_actions[0].cpu(),
                'expert_raw': expert_raw[0].cpu(),
                'action_mask': action_mask[0].cpu(),
                'log_prob': log_prob[0].cpu(),
            }

        self._update_stats(result)
        return result

    # ------------------------------------------------------------------ stats

    def _update_stats(self, result: Dict):
        """Update inference statistics."""
        s = self.stats
        s['total_queries'] += 1
        s['bypasses'] += result['path'] == 'bypass'
        s['episodic_accesses'] += result['memory_used']['episodic']
        s['semantic_accesses'] += result['memory_used']['semantic']
        s['verifications'] += result['verified']
        s['abstentions'] += result['abstained']
        s['latencies'].append(result['latency'])
        s['compute_units'].append(result['compute_units'])

    def get_stats(self) -> Dict:
        """Inference statistics: rates, p50/p95 latency and mean compute."""
        stats = {k: v for k, v in self.stats.items() if k not in ('latencies', 'compute_units')}
        n = stats['total_queries']
        if n > 0:
            for key, rate in (('bypasses', 'bypass_rate'), ('episodic_accesses', 'episodic_access_rate'),
                              ('semantic_accesses', 'semantic_access_rate'), ('verifications', 'verification_rate'),
                              ('abstentions', 'abstention_rate')):
                stats[rate] = stats[key] / n
            latencies = sorted(self.stats['latencies'])
            stats['latency_p50'] = latencies[len(latencies) // 2]
            stats['latency_p95'] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            stats['mean_compute_units'] = sum(self.stats['compute_units']) / n
        return stats

    def reset_stats(self):
        """Reset inference statistics."""
        self.stats = {
            'total_queries': 0,
            'bypasses': 0,
            'episodic_accesses': 0,
            'semantic_accesses': 0,
            'verifications': 0,
            'abstentions': 0,
            'latencies': [],
            'compute_units': [],
        }
