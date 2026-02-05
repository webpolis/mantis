"""
HMST Inference Engine

Orchestrates dynamic routing and generation with memory systems.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import time


class HMSTInferenceEngine:
    """
    Main inference engine for HMST.

    Coordinates:
    - Meta-controller routing decisions
    - Memory retrieval (episodic + semantic)
    - Base model generation with expert routing
    - Critic verification
    """

    def __init__(
        self,
        base_model,
        meta_controller,
        episodic_memory,
        semantic_memory,
        critic_model,
        state_encoder,
        tokenizer=None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.base = base_model.to(device)
        self.meta = meta_controller.to(device)
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.critic = critic_model.to(device) if critic_model else None
        self.state_encoder = state_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Inference statistics
        self.stats = {
            'total_queries': 0,
            'early_exits': 0,
            'episodic_accesses': 0,
            'semantic_accesses': 0,
            'verifications': 0,
            'avg_latency': 0.0
        }

    @torch.no_grad()
    def generate(
        self,
        query: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_details: bool = False
    ) -> Dict:
        """
        Generate response with dynamic routing.

        Args:
            query: Input query string
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_details: Return detailed routing information

        Returns:
            Dict with response and metadata
        """
        start_time = time.time()

        # Step 1: Tokenize and encode query
        query_tokens = self._tokenize(query)
        query_emb = self.base.encode(query_tokens)  # (1, d_model)

        # Step 2: Quick forward pass for uncertainty estimation
        with torch.no_grad():
            output = self.base(query_tokens, return_hidden=True)
            logits = output['logits']
            hidden = output['last_hidden']

        uncertainty = self._estimate_uncertainty(logits)

        # Step 3: Create state summary for meta-controller
        state_summary = self._create_state_summary(
            uncertainty,
            context_length=query_tokens.size(1),
            confidence=0.5  # Placeholder
        )

        # Step 4: Meta-controller routing decisions
        routing_continuous = self.meta(query_emb, state_summary)
        routing_decisions = self.meta.decide(routing_continuous, threshold=0.5)

        # Step 5: Early exit check
        if routing_decisions['early_exit'] and uncertainty < 0.2:
            response = self._generate_simple(query_tokens, max_length, temperature, top_p)

            result = {
                'response': response,
                'path': 'early_exit',
                'uncertainty': uncertainty,
                'verified': False,
                'latency': time.time() - start_time
            }

            self._update_stats(result)
            return result

        # Step 6: Memory retrieval
        context_parts = [query]
        memory_used = {'episodic': False, 'semantic': False}
        facts = []

        if routing_decisions['episodic']:
            recent_memories = self.episodic.retrieve(query_emb.squeeze(0), top_k=3)
            if recent_memories:
                context_parts.append("[Recent context]: " + " | ".join(recent_memories))
                memory_used['episodic'] = True

        if routing_decisions['semantic']:
            facts = self.semantic.retrieve(query_emb.squeeze(0), top_k=5)
            if facts:
                context_parts.append("[Relevant facts]: " + " | ".join(facts))
                memory_used['semantic'] = True

        # Combine context
        augmented_context = "\n\n".join(context_parts)
        context_tokens = self._tokenize(augmented_context)

        # Step 7: Generate with expert routing
        expert_weights = routing_decisions['expert_weights']

        response = self._generate_with_experts(
            context_tokens,
            max_length,
            expert_weights,
            temperature,
            top_p
        )

        # Step 8: Verification (if needed)
        verified = False
        confidence = 0.0

        if (routing_decisions['verification'] or uncertainty > 0.5) and self.critic:
            response_tokens = self._tokenize(response)
            fact_tokens = self._tokenize(" ".join(facts) if facts else "")

            confidence = self.critic.verify(
                query_tokens,
                response_tokens,
                fact_tokens if fact_tokens.size(1) > 0 else None
            )
            verified = True

            # Low confidence fallback
            if confidence < 0.6:
                response = self._generate_conservative(query, augmented_context)

        # Final result
        result = {
            'response': response,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'path': 'full',
            'memory_used': memory_used,
            'verified': verified,
            'expert_weights': expert_weights.tolist() if torch.is_tensor(expert_weights) else expert_weights,
            'latency': time.time() - start_time
        }

        if return_details:
            result['routing_decisions'] = routing_decisions
            result['routing_continuous'] = {k: v.tolist() if torch.is_tensor(v) else v
                                           for k, v in routing_continuous.items()}

        self._update_stats(result)

        return result

    def _tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text to tensor.

        Returns:
            (1, seq_len) token ids
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, return_tensors='pt')
        else:
            # Placeholder: random tokens
            tokens = torch.randint(0, 128000, (1, min(len(text.split()), 100)))

        return tokens.to(self.device)

    def _detokenize(self, tokens: torch.Tensor) -> str:
        """
        Convert tokens back to text.
        """
        if self.tokenizer:
            return self.tokenizer.decode(tokens.squeeze(0), skip_special_tokens=True)
        else:
            return f"<generated_text_{tokens.size(1)}_tokens>"

    def _estimate_uncertainty(self, logits: torch.Tensor) -> float:
        """
        Compute uncertainty from output distribution.

        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            Scalar uncertainty in [0, 1]
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float32))
        normalized_entropy = (entropy / max_entropy).mean()

        return normalized_entropy.item()

    def _create_state_summary(
        self,
        uncertainty: float,
        context_length: int,
        confidence: float
    ) -> torch.Tensor:
        """
        Create state summary for meta-controller.

        Returns:
            (1, state_dim) state vector
        """
        # Prepare inputs
        uncertainty_t = torch.tensor([[uncertainty]], device=self.device)
        context_t = torch.tensor([[context_length / 8192.0]], device=self.device)  # Normalize
        memory_t = torch.tensor([[0.0, 0.0]], device=self.device)  # No memory accessed yet
        confidence_t = torch.tensor([[confidence]], device=self.device)

        # Encode
        state = self.state_encoder(uncertainty_t, context_t, memory_t, confidence_t)

        return state

    def _generate_simple(
        self,
        input_tokens: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float
    ) -> str:
        """
        Simple generation without expert routing (early exit path).
        """
        # Greedy or sampling decode
        generated = self._decode(input_tokens, max_length, temperature, top_p)
        return self._detokenize(generated)

    def _generate_with_experts(
        self,
        input_tokens: torch.Tensor,
        max_length: int,
        expert_weights: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> str:
        """
        Generation with meta-controller specified expert routing.
        """
        # Forward with expert weights
        generated = self._decode(
            input_tokens,
            max_length,
            temperature,
            top_p,
            expert_weights=expert_weights
        )

        return self._detokenize(generated)

    def _generate_conservative(self, query: str, context: str) -> str:
        """
        Conservative generation when confidence is low.

        Uses retrieval-based fallback or abstention.
        """
        return f"I don't have sufficient confidence to answer '{query}' accurately. " \
               f"Would you like me to provide sources instead?"

    def _decode(
        self,
        input_tokens: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        expert_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive decoding.

        Args:
            input_tokens: (1, seq_len)
            max_length: Maximum new tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling
            expert_weights: Optional expert routing

        Returns:
            (1, seq_len + generated_len) full sequence
        """
        generated = input_tokens.clone()

        for _ in range(max_length):
            # Forward pass
            output = self.base(generated, expert_weights=expert_weights)
            logits = output['logits'][:, -1, :]  # (1, vocab_size)

            # Sample next token
            next_token = self._sample(logits, temperature, top_p)

            # Append
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Stop if EOS (assuming token 2 is EOS)
            if next_token.item() == 2:
                break

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """
        Sample next token with temperature and nucleus sampling.

        Args:
            logits: (1, vocab_size)
            temperature: Temperature parameter
            top_p: Nucleus probability mass

        Returns:
            (1,) sampled token
        """
        # Squeeze to 1D for easier indexing
        logits = logits.squeeze(0)  # (vocab_size,)

        # Apply temperature
        logits = logits / temperature

        # Nucleus sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        # Zero out removed tokens
        probs[sorted_indices[sorted_indices_to_remove]] = 0
        probs = probs / probs.sum()

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def _update_stats(self, result: Dict):
        """Update inference statistics."""
        self.stats['total_queries'] += 1

        if result['path'] == 'early_exit':
            self.stats['early_exits'] += 1

        if result['memory_used']['episodic']:
            self.stats['episodic_accesses'] += 1

        if result['memory_used']['semantic']:
            self.stats['semantic_accesses'] += 1

        if result['verified']:
            self.stats['verifications'] += 1

        # Running average latency
        alpha = 0.1
        self.stats['avg_latency'] = (
            alpha * result['latency'] +
            (1 - alpha) * self.stats['avg_latency']
        )

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
