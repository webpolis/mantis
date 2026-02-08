"""
Mixture-of-Experts Base Model for HMST Architecture

Implements sparse MoE transformer with top-k routing and load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class Expert(nn.Module):
    """Single expert network (feedforward)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with top-k routing and load balancing.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

        # Gating network
        self.gate = nn.Linear(d_model, n_experts)

        # Track expert load for balancing
        self.register_buffer('expert_counts', torch.zeros(n_experts))

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with top-k expert routing.

        Args:
            x: (batch, seq_len, d_model)
            expert_weights: Optional (batch, n_experts) from meta-controller

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: Dict with load balancing loss
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        # Compute gate scores
        gate_logits = self.gate(x_flat)  # (batch * seq_len, n_experts)
        if expert_weights is not None:
            # Bias the learned gate with meta-controller signal
            bias = expert_weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, self.n_experts)
            gate_probs = F.softmax(gate_logits + bias, dim=-1)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute load balancing loss
        load_loss = self._compute_load_balance_loss(gate_probs)

        # Route tokens to experts
        output = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i].unsqueeze(-1)

            # Group by expert for efficiency
            for expert_id in range(self.n_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    tokens = x_flat[mask]
                    expert_out = self.experts[expert_id](tokens)
                    output[mask] += expert_prob[mask] * expert_out

        # Reshape back
        output = output.view(batch_size, seq_len, d_model)

        aux_info = {
            'load_balance_loss': load_loss,
            'gate_probs': gate_probs.view(batch_size, seq_len, self.n_experts),
            'expert_usage': top_k_indices.view(batch_size, seq_len, self.top_k)
        }

        return output, aux_info

    def _compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.

        L = N * Σ_i (f_i * P_i)
        where f_i = fraction of tokens dispatched to expert i (hard routing)
              P_i = mean routing probability for expert i (soft)
              N = number of experts
        """
        # Hard dispatch fractions: which expert gets each token (top-1)
        top1_indices = gate_probs.argmax(dim=-1)  # (num_tokens,)
        f = torch.zeros(self.n_experts, device=gate_probs.device)
        for i in range(self.n_experts):
            f[i] = (top1_indices == i).float().mean()

        # Soft routing probabilities
        P = gate_probs.mean(dim=0)  # (n_experts,)

        balance_loss = self.n_experts * (f * P).sum()

        return self.load_balance_weight * balance_loss


class TransformerBlock(nn.Module):
    """Transformer block with MoE feedforward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_moe: bool = True
    ):
        super().__init__()
        self.use_moe = use_moe

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Feedforward (MoE or standard)
        if use_moe:
            self.ff = MoELayer(d_model, d_ff, n_experts, top_k, dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )

        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: Causal mask - float with -inf for masked positions
            key_padding_mask: Padding mask (batch, seq_len) - bool (True=ignore padding)
            expert_weights: Optional routing weights from meta-controller
            past_kv: Optional cached (key, value) from previous steps

        Returns:
            output: (batch, seq_len, d_model)
            aux_info: None or dict with MoE losses
            present_kv: (key, value) cache for this layer
        """
        # Self-attention with pre-norm
        normed = self.attn_norm(x)

        if past_kv is not None:
            past_key, past_value = past_kv
            # Concat past keys/values with current for full context
            key = torch.cat([past_key, normed], dim=1)
            value = torch.cat([past_value, normed], dim=1)
        else:
            key = normed
            value = normed

        # Store current kv for cache
        present_kv = (key.detach(), value.detach())

        attn_out, _ = self.attn(
            normed, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)

        # Feedforward
        normed = self.ff_norm(x)

        if self.use_moe:
            ff_out, aux_info = self.ff(normed, expert_weights)
            x = x + self.dropout(ff_out)
            return x, aux_info, present_kv
        else:
            ff_out = self.ff(normed)
            x = x + ff_out
            return x, None, present_kv


class BaseMoEModel(nn.Module):
    """
    Base MoE Transformer model for HMST.

    12B total parameters, 2B active per forward pass.
    """

    def __init__(
        self,
        vocab_size: int = 128000,
        d_model: int = 2048,
        n_layers: int = 24,
        n_heads: int = 32,
        d_ff: int = 8192,
        n_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.load_balance_weight = load_balance_weight
        self.gradient_checkpointing = False

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_experts, top_k, dropout, use_moe=True)
            for _ in range(n_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with GPT-style standards."""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # GPT-style initialization: Normal(0, 0.02)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

        self.apply(_init_module)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    @staticmethod
    def _checkpoint_forward(layer, x, causal_mask, key_padding_mask, expert_weights):
        """Wrapper for gradient checkpointing (no KV-cache during training)."""
        return layer(x, causal_mask, key_padding_mask, expert_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        past_key_values: Optional[list] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: Optional (batch, seq_len)
            expert_weights: Optional (batch, n_experts) from meta-controller
            return_hidden: Return hidden states
            past_key_values: List of (key, value) tuples per layer for KV-cache
            use_cache: Whether to return present key values for caching

        Returns:
            Dict with logits, losses, optional hidden states, and optional present_key_values
        """
        batch_size, seq_len = input_ids.shape

        # Position offset for KV-cache (positions start after cached tokens)
        past_len = past_key_values[0][0].size(1) if past_key_values else 0

        # Embeddings
        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.clamp(max=self.max_seq_len - 1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Causal mask: only needed for the new tokens attending to full context
        total_len = past_len + seq_len
        causal_mask = torch.triu(
            torch.full((seq_len, total_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1 + past_len
        )

        # Prepare padding mask (if provided)
        # nn.MultiheadAttention expects True for positions to IGNORE
        key_padding_mask = None
        if attention_mask is not None:
            # Input format: 1=valid token, 0=padding
            key_padding_mask = (attention_mask == 0)

        # Forward through layers
        total_load_loss = 0.0
        hidden_states = [] if return_hidden else None
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x, aux_info, present_kv = torch.utils.checkpoint.checkpoint(
                    self._checkpoint_forward,
                    layer, x, causal_mask, key_padding_mask, expert_weights,
                    use_reentrant=False
                )
            else:
                x, aux_info, present_kv = layer(x, causal_mask, key_padding_mask, expert_weights, past_kv=layer_past)

            if aux_info is not None:
                total_load_loss += aux_info['load_balance_loss']

            if return_hidden:
                hidden_states.append(x)

            if use_cache:
                present_key_values.append(present_kv)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {
            'logits': logits,
            'load_balance_loss': total_load_loss,
        }

        if return_hidden:
            output['hidden_states'] = hidden_states
            output['last_hidden'] = x

        if use_cache:
            output['past_key_values'] = present_key_values

        return output

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for input (for meta-controller).

        Returns:
            (batch, d_model) pooled representation
        """
        output = self.forward(input_ids, return_hidden=True)
        last_hidden = output['last_hidden']  # (batch, seq_len, d_model)

        # Mean pooling
        pooled = last_hidden.mean(dim=1)
        return pooled

    def count_parameters(self) -> Dict[str, int]:
        """Count total and active parameters."""
        total = sum(p.numel() for p in self.parameters())

        # Estimate active (rough: top-k/n_experts of FFN params)
        moe_params = sum(
            p.numel() for name, p in self.named_parameters()
            if 'experts' in name
        )
        non_moe_params = total - moe_params

        # Approximate active: all non-MoE + top_k/n_experts * MoE
        active_ratio = 2 / 8  # top-2 out of 8 experts
        active = non_moe_params + int(moe_params * active_ratio)

        return {
            'total': total,
            'active': active,
            'moe_params': moe_params,
            'non_moe_params': non_moe_params
        }
