"""
MANTIS Inference Engine

Orchestrates dynamic routing and generation with memory systems.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch

from mantis.configs.model_config import InferenceConfig
from mantis.inference.generation import generate_tokens
from mantis.models.meta_controller import GATES, MetaController, StateSummaryEncoder


def build_policy(config) -> Tuple[MetaController, StateSummaryEncoder]:
    """Meta-controller and state encoder sized from a MANTISConfig."""
    mc = config.meta_controller
    meta = MetaController(
        d_model=config.base_moe.d_model,
        n_layers=mc.n_layers,
        d_ff=mc.d_ff,
        dropout=mc.dropout,
        n_experts=config.base_moe.n_experts,
        state_dim=mc.state_dim,
    )
    return meta, StateSummaryEncoder(state_dim=mc.state_dim)


class MANTISInferenceEngine:
    """
    Main inference engine for MANTIS.

    Coordinates:
    - Meta-controller routing decisions
    - Memory retrieval (episodic + semantic) and episodic writes
    - Base model generation with expert routing
    - Critic verification

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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        self.gate_mask = torch.tensor(
            [[1.0, float(episodic_memory is not None), float(semantic_memory is not None),
              float(critic_model is not None)]],
            device=device,
        )
        self.banned_ids = tokenizer.non_generable_ids
        counts = self.base.count_parameters()
        self.active_param_ratio = counts['active'] / counts['total']

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
    ) -> 'MANTISInferenceEngine':
        """
        Build the full engine from saved artifacts.

        A Stage 3 policy checkpoint records the paths of the components it was
        trained with; explicit arguments override them.
        """
        from mantis.utils.checkpoints import check_tokenizer, compat_load, load_base_model

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        policy = compat_load(policy_checkpoint) if policy_checkpoint else {}
        base_checkpoint = base_checkpoint or policy.get('base_checkpoint')
        memory_checkpoint = memory_checkpoint or policy.get('memory_checkpoint')
        semantic_store = semantic_store or policy.get('semantic_store')
        critic_checkpoint = critic_checkpoint or policy.get('critic_checkpoint')
        if not base_checkpoint:
            raise ValueError("A base model checkpoint is required")

        base, tokenizer, base_ckpt = load_base_model(base_checkpoint, device, tokenizer_path)
        config = base_ckpt['config']

        meta, state_encoder = build_policy(config)
        if policy:
            check_tokenizer(policy, tokenizer, policy_checkpoint)
            meta.load_state_dict(policy['meta_controller_state_dict'])
            state_encoder.load_state_dict(policy['state_encoder_state_dict'])
        else:
            print("⚠️  No policy checkpoint: meta-controller is untrained")

        episodic = None
        if memory_checkpoint:
            from mantis.memory.episodic import EpisodicMemory
            from mantis.models.ssm import EpisodicMemorySSM
            mem = compat_load(memory_checkpoint)
            check_tokenizer(mem, tokenizer, memory_checkpoint)
            em = config.episodic_memory
            ssm = EpisodicMemorySSM(d_model=config.base_moe.d_model, d_state=em.d_state, n_blocks=em.n_blocks,
                                    d_conv=em.d_conv, expand=em.expand, max_seq_len=em.max_seq_len,
                                    dropout=em.dropout)
            ssm.load_state_dict(mem['episodic_ssm_state_dict'])
            episodic = EpisodicMemory(ssm, max_entries=em.max_entries, context_window=em.max_seq_len, device=device)

        semantic = None
        if semantic_store:
            from mantis.memory.semantic import SemanticMemory
            semantic = SemanticMemory.load(semantic_store, use_gpu=config.semantic_memory.use_gpu)

        critic = None
        if critic_checkpoint:
            from mantis.models.critic import CriticModel
            crit = compat_load(critic_checkpoint)
            check_tokenizer(crit, tokenizer, critic_checkpoint)
            critic = CriticModel(**vars(crit['config'].critic))
            critic.load_state_dict(crit['critic_state_dict'])

        engine = cls(base, meta, state_encoder, tokenizer, episodic, semantic, critic,
                     config=config.inference, device=device)
        engine.mantis_config = config
        return engine

    # ----------------------------------------------------------------- policy

    def _state_features(self, logits: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, float]:
        """
        Features for the state encoder, as one (1, 5) row:
        [uncertainty, context fill, episodic fill, semantic non-empty, confidence].

        Uncertainty is the mean normalized next-token entropy over the query;
        confidence is the mean top-1 probability.
        """
        probs = torch.softmax(logits[0].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        uncertainty = float((entropy / torch.log(torch.tensor(float(probs.size(-1))))).mean())
        confidence = float(probs.max(dim=-1).values.mean())
        episodic_fill = self.episodic.size() / self.episodic.max_entries if self.episodic else 0.0
        semantic_ready = float(bool(self.semantic and self.semantic.size() > 0))
        features = torch.tensor(
            [[uncertainty, seq_len / self.base.max_seq_len, episodic_fill, semantic_ready, confidence]],
            device=self.device,
        )
        return features, uncertainty

    def encode_state(self, features: torch.Tensor) -> torch.Tensor:
        """(batch, 5) features -> (batch, state_dim) summary."""
        return self.state_encoder(features[:, 0:1], features[:, 1:2], features[:, 2:4], features[:, 4:5])

    # ------------------------------------------------------------- generation

    def _decode(self, input_ids: List[int], max_length: int, temperature: float, top_p: float,
                expert_weights: Optional[torch.Tensor] = None) -> Tuple[List[int], List[float]]:
        """Generate new tokens (EOS excluded) and their log-probabilities."""
        tokens, log_probs = [], []
        for token, log_prob in generate_tokens(
            self.base, input_ids, max_length, temperature=temperature, top_p=top_p,
            banned_ids=self.banned_ids, stop_ids=[self.tokenizer.eos_token_id],
            expert_weights=expert_weights,
        ):
            if token == self.tokenizer.eos_token_id:
                break
            tokens.append(token)
            log_probs.append(log_prob)
        return tokens, log_probs

    def _with_memory(self, memory_parts: List[str], query_ids: List[int], max_length: int) -> List[int]:
        """
        Prompt = retrieved memory, then the query. Memory is cut so the query and
        up to half a window of generation still fit in the attention window.
        """
        if not memory_parts:
            return query_ids
        reserve = min(max_length, self.base.max_seq_len // 2)
        budget = max(0, self.base.max_seq_len - len(query_ids) - reserve)
        memory_ids = self.tokenizer.encode("\n\n".join(memory_parts) + "\n\n")[:budget]
        return memory_ids + query_ids

    def _remember(self, token_ids: List[int]) -> None:
        """Store an interaction (query + response) in episodic memory."""
        token_ids = token_ids[-self.base.max_seq_len:]
        ids = torch.tensor([token_ids], device=self.device)
        hidden = self.base(ids, return_hidden=True)['last_hidden'][0]
        self.episodic.add(ids[0], hidden)

    @torch.no_grad()
    def generate(
        self,
        query: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        explore: bool = False,
        return_details: bool = False
    ) -> Dict:
        """
        Generate a response with dynamic routing.

        Args:
            query: Input query string
            max_length: Maximum new tokens (default: config)
            temperature: Sampling temperature (default: config)
            top_p: Nucleus sampling parameter (default: config)
            explore: Sample routing actions from the policy (RL rollouts)
            return_details: Include routing probabilities and the rollout record

        Returns:
            Dict with the response (completion only) and metadata
        """
        cfg = self.config
        max_length = cfg.max_length if max_length is None else max_length
        temperature = cfg.temperature if temperature is None else temperature
        top_p = cfg.top_p if top_p is None else top_p
        start_time = time.time()

        query_ids = self.tokenizer.encode(query)[-self.base.max_seq_len:]
        if not query_ids:
            raise ValueError("Query is empty")
        output = self.base(torch.tensor([query_ids], device=self.device), return_hidden=True)
        hidden = output['last_hidden']
        query_emb = hidden.mean(dim=1)

        features, uncertainty = self._state_features(output['logits'], len(query_ids))
        decisions = self.meta(query_emb, self.encode_state(features))
        gate_actions, expert_bias = self.meta.act(
            decisions, self.gate_mask, sample=explore, threshold=cfg.routing_threshold
        )
        log_prob = self.meta.log_prob(decisions, gate_actions, expert_bias, self.gate_mask)
        gates = {gate: bool(v) for gate, v in zip(GATES, gate_actions[0].tolist())}

        memory_used = {'episodic': False, 'semantic': False}
        critic_score = None
        facts: List[str] = []

        if gates['early_exit'] and uncertainty < cfg.early_exit_uncertainty_threshold:
            path = 'early_exit'
            response_ids, log_probs = self._decode(query_ids, max_length, temperature, top_p)
        else:
            path = 'full'
            memory_parts = []
            if gates['episodic']:
                memories = [self.tokenizer.decode(t) for t in self.episodic.retrieve(hidden[0], top_k=3)]
                if memories:
                    memory_parts.append("[Recent context]: " + " | ".join(memories))
                    memory_used['episodic'] = True
            if gates['semantic']:
                facts = self.semantic.retrieve(query_emb[0], top_k=5)
                if facts:
                    memory_parts.append("[Relevant facts]: " + " | ".join(facts))
                    memory_used['semantic'] = True

            context_ids = self._with_memory(memory_parts, query_ids, max_length)
            response_ids, log_probs = self._decode(context_ids, max_length, temperature, top_p, expert_bias)

        response = self.tokenizer.decode(response_ids)
        abstained = False
        if path == 'full' and gates['verification']:
            facts_ids = self.tokenizer.encode(" ".join(facts)) if facts else None
            critic_score = self.critic.verify(query_ids, response_ids, facts_ids)
            if critic_score < cfg.verification_confidence_threshold:
                response = self._generate_conservative(query)
                abstained = True

        # Remember what was actually returned, never a completion the critic rejected
        if self.episodic is not None:
            self._remember(query_ids + self.tokenizer.encode(response))

        confidence = (
            float(torch.tensor(log_probs).mean().exp()) if log_probs else 0.0
        )
        result = {
            'response': response,
            'path': path,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'critic_score': critic_score,
            'abstained': abstained,
            'verified': critic_score is not None,
            'memory_used': memory_used,
            'expert_weights': expert_bias[0].tolist(),
            'num_tokens': len(response_ids),
            'latency': time.time() - start_time,
        }

        if return_details:
            result['routing_decisions'] = gates
            result['routing_probs'] = {gate: float(decisions[gate]) for gate in GATES}
            result['rollout'] = {
                'query_emb': query_emb[0].cpu(),
                'features': features[0].cpu(),
                'gate_actions': gate_actions[0].cpu(),
                'expert_bias': expert_bias[0].cpu(),
                'gate_mask': self.gate_mask[0].cpu(),
                'log_prob': log_prob[0].cpu(),
            }

        self._update_stats(result)
        return result

    def _generate_conservative(self, query: str) -> str:
        """Abstention used when the critic rejects a response."""
        return f"I don't have sufficient confidence to answer '{query}' accurately."

    # ------------------------------------------------------------------ stats

    def _update_stats(self, result: Dict):
        """Update inference statistics."""
        self.stats['total_queries'] += 1
        self.stats['early_exits'] += result['path'] == 'early_exit'
        self.stats['episodic_accesses'] += result['memory_used']['episodic']
        self.stats['semantic_accesses'] += result['memory_used']['semantic']
        self.stats['verifications'] += result['verified']

        # Running average latency
        alpha = 0.1
        self.stats['avg_latency'] = alpha * result['latency'] + (1 - alpha) * self.stats['avg_latency']

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        stats = self.stats.copy()

        if stats['total_queries'] > 0:
            stats['early_exit_rate'] = stats['early_exits'] / stats['total_queries']
            stats['episodic_access_rate'] = stats['episodic_accesses'] / stats['total_queries']
            stats['semantic_access_rate'] = stats['semantic_accesses'] / stats['total_queries']
            stats['verification_rate'] = stats['verifications'] / stats['total_queries']

        return stats

    def reset_stats(self):
        """Reset inference statistics."""
        self.stats = {
            'total_queries': 0,
            'early_exits': 0,
            'episodic_accesses': 0,
            'semantic_accesses': 0,
            'verifications': 0,
            'avg_latency': 0.0
        }
