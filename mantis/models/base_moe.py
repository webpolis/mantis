"""
Mixture-of-Experts Base Model for MANTIS Architecture

Implements a sparse MoE transformer with top-k routing, load balancing,
rotary positional embeddings and a projected key/value cache.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Tuple, Optional, Dict, List

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (GPT-NeoX half-split layout)."""

    def __init__(self, d_head: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq)  # (seq, d_head / 2)
        return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (batch, heads, seq, d_head) by per-position angles."""
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and a (batch, heads, seq, d_head) KV cache."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, d_model = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch, seq_len, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        cos, sin = rope
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(out), ((k, v) if use_cache else None)


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

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with top-k expert routing.

        Args:
            x: (batch, seq_len, d_model)
            expert_weights: Optional (batch, n_experts) raw logits from the
                meta-controller, added as a bias to the learned gate logits

        Returns:
            output: (batch, seq_len, d_model)
            load_balance_loss: scalar
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        gate_logits = self.gate(x_flat).float()
        if expert_weights is not None:
            bias = expert_weights.float().unsqueeze(1).expand(-1, seq_len, -1)
            gate_logits = gate_logits + bias.reshape(-1, self.n_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot = (top_k_indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(x_flat[token_idx]) * top_k_probs[token_idx, slot].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_out.to(output.dtype))

        return output.view(batch_size, seq_len, d_model), self._compute_load_balance_loss(gate_probs)

    def _compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.

        L = N * Σ_i (f_i * P_i)
        where f_i = fraction of tokens dispatched to expert i (hard routing)
              P_i = mean routing probability for expert i (soft)
              N = number of experts
        """
        top1 = gate_probs.argmax(dim=-1)
        f = torch.bincount(top1, minlength=self.n_experts).float() / top1.numel()
        P = gate_probs.mean(dim=0)
        return self.load_balance_weight * self.n_experts * (f * P).sum()


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with an MoE (or dense) feedforward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        use_moe: bool = True
    ):
        super().__init__()
        self.use_moe = use_moe

        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        if use_moe:
            self.ff = MoELayer(d_model, d_ff, n_experts, top_k, dropout, load_balance_weight)
        else:
            self.ff = Expert(d_model, d_ff, dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        Returns:
            output: (batch, seq_len, d_model)
            load_balance_loss: scalar tensor, or None for dense blocks
            present_kv: (key, value) each (batch, heads, total_len, d_head), or None
        """
        attn_out, present_kv = self.attn(self.attn_norm(x), rope, attn_mask, past_kv, use_cache)
        x = x + self.dropout(attn_out)

        normed = self.ff_norm(x)
        if self.use_moe:
            ff_out, load_loss = self.ff(normed, expert_weights)
        else:
            ff_out, load_loss = self.ff(normed), None
        x = x + self.dropout(ff_out)
        return x, load_loss, present_kv


class BaseMoEModel(nn.Module):
    """
    Base MoE Transformer model for MANTIS.

    `max_seq_len` is the attention window: training sequences should not exceed
    it, and generation re-encodes the most recent tokens to stay within it.
    """

    def __init__(
        self,
        vocab_size: int = 512,
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
        self.n_experts = n_experts
        self.top_k = top_k
        self.gradient_checkpointing = False

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.dropout = nn.Dropout(dropout)

        use_moe = n_experts > 1
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_experts, top_k, dropout,
                             load_balance_weight, use_moe=use_moe)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    @classmethod
    def from_config(cls, config) -> 'BaseMoEModel':
        """Build from a BaseMoEConfig."""
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            top_k=config.top_k,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            load_balance_weight=config.load_balance_weight,
        )

    def _init_weights(self):
        """Initialize weights with GPT-style standards."""
        def _init_module(module):
            if isinstance(module, nn.Linear):
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
    def cache_len(past_key_values: Optional[KVCache]) -> int:
        return past_key_values[0][0].size(2) if past_key_values else 0

    def _attention_mask(
        self,
        seq_len: int,
        past_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Boolean mask (True = attend). None means plain causal attention."""
        if attention_mask is None and past_len == 0:
            return None
        if attention_mask is not None and past_len > 0:
            raise ValueError("attention_mask is not supported together with a KV cache")

        q_pos = torch.arange(seq_len, device=device).unsqueeze(1) + past_len
        k_pos = torch.arange(past_len + seq_len, device=device).unsqueeze(0)
        allowed = k_pos <= q_pos  # (seq, total)
        if attention_mask is None:
            return allowed

        valid = attention_mask.bool()[:, None, None, :]  # (batch, 1, 1, seq)
        # Every query keeps itself visible so fully padded rows never produce NaN.
        diagonal = torch.eye(seq_len, dtype=torch.bool, device=device)
        return (allowed & valid) | diagonal

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: Optional (batch, seq_len), 1 = token, 0 = padding.
                Not needed for right-padded batches (causal attention already
                keeps real tokens from seeing later padding).
            expert_weights: Optional (batch, n_experts) from meta-controller
            return_hidden: Return hidden states
            past_key_values: Per-layer (key, value) cache from a previous call
            use_cache: Return the updated cache as `past_key_values`

        Returns:
            Dict with logits, load_balance_loss, optional hidden states and cache
        """
        batch_size, seq_len = input_ids.shape
        past_len = self.cache_len(past_key_values)

        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        rope = self.rotary(positions)
        attn_mask = self._attention_mask(seq_len, past_len, attention_mask, input_ids.device)

        x = self.dropout(self.token_embedding(input_ids))

        total_load_loss = x.new_zeros(())
        hidden_states = [] if return_hidden else None
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                x, load_loss, present_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, rope, attn_mask, expert_weights, None, False,
                    use_reentrant=False
                )
            else:
                x, load_loss, present_kv = layer(x, rope, attn_mask, expert_weights, layer_past, use_cache)

            if load_loss is not None:
                total_load_loss = total_load_loss + load_loss
            if return_hidden:
                hidden_states.append(x)
            if use_cache:
                present_key_values.append(present_kv)

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

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mean-pooled final hidden state (for meta-controller and memory).

        Args:
            input_ids: (batch, seq_len), right-padded if batched
            attention_mask: Optional (batch, seq_len); padding is excluded from the mean

        Returns:
            (batch, d_model) pooled representation
        """
        last_hidden = self.forward(input_ids, return_hidden=True)['last_hidden']
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and active (per-token) parameters."""
        total = sum(p.numel() for p in self.parameters())
        moe_params = sum(
            p.numel() for name, p in self.named_parameters()
            if '.experts.' in name
        )
        non_moe_params = total - moe_params
        active = non_moe_params + moe_params * self.top_k // self.n_experts

        return {
            'total': total,
            'active': active,
            'moe_params': moe_params,
            'non_moe_params': non_moe_params
        }
