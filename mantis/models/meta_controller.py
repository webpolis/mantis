"""
Meta-Controller for MANTIS Architecture

Residual MLP policy that outputs routing decisions for dynamic query processing.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from typing import Dict, Tuple

GATES = ('bypass', 'episodic', 'semantic', 'verification')
EXPERT = len(GATES)  # index of the expert-bias flag in an action mask


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual feedforward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class MetaController(nn.Module):
    """
    Routing policy over a pooled query embedding plus a state summary.

    The meta-controller reads the query and outputs 5 routing decisions:
    - Bypass: skip the optional components (memory reads, expert bias,
      verification). The backbone still runs at full depth.
    - Episodic Access: Query recent interaction history
    - Semantic Retrieval: Access long-term knowledge base
    - Expert Bias: bounded, layer-specific additive bias on the MoE gate logits
    - Verification: Trigger critic model for fact-checking

    The input is one vector per query, so the network is a stack of residual
    MLP blocks. As a policy, gates are Bernoulli and the raw expert bias is a
    diagonal Gaussian around `expert_raw`; the applied bias is
    `expert_bias_scale * tanh(raw)`, zero at initialization, so it cannot
    overwhelm the pretrained gate logits. Deterministic routing thresholds the
    gate probabilities and uses the Gaussian mean.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 6,
        d_ff: int = 4096,
        dropout: float = 0.1,
        n_experts: int = 8,
        n_moe_layers: int = 24,
        state_dim: int = 128,
        expert_bias_scale: float = 2.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.n_moe_layers = n_moe_layers
        self.state_dim = state_dim
        self.expert_bias_scale = expert_bias_scale

        self.embedding = nn.Linear(d_model + state_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualMLPBlock(d_model, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.gate_head = nn.Linear(d_model, len(GATES))
        self.expert_selector = nn.Linear(d_model, n_moe_layers * n_experts)
        self.expert_log_std = nn.Parameter(torch.full((n_moe_layers * n_experts,), -1.0))

        self._init_weights()

    def _init_weights(self):
        """Xavier init everywhere; the expert selector starts at zero (no routing shift)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.expert_selector.weight)

    def forward(
        self,
        query_embedding: torch.Tensor,
        state_summary: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing distributions.

        Args:
            query_embedding: (batch, d_model) - Encoded query
            state_summary: (batch, state_dim) - Confidence scores, context indicators

        Returns:
            Dict with:
                - bypass / episodic / semantic / verification: (batch, 1) probabilities
                - gate_logits: (batch, 4) logits of those gates, in GATES order
                - expert_raw: (batch, n_moe_layers * n_experts) Gaussian mean of the raw bias
        """
        x = self.embedding(torch.cat([query_embedding, state_summary], dim=-1))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        h = self.final_norm(x)

        gate_logits = self.gate_head(h)
        decisions = {gate: torch.sigmoid(gate_logits[:, i:i + 1]) for i, gate in enumerate(GATES)}
        decisions['gate_logits'] = gate_logits
        decisions['expert_raw'] = self.expert_selector(h)
        return decisions

    def expert_bias(self, expert_raw: torch.Tensor) -> torch.Tensor:
        """(batch, n_moe_layers * n_experts) raw -> (batch, n_moe_layers, n_experts) bounded bias."""
        bias = self.expert_bias_scale * torch.tanh(expert_raw)
        return bias.view(-1, self.n_moe_layers, self.n_experts)

    def act(
        self,
        decisions: Dict[str, torch.Tensor],
        gate_mask: torch.Tensor,
        sample: bool,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Choose routing actions.

        Args:
            decisions: Output of forward()
            gate_mask: (batch, 4) 1 where a gate's component exists; masked gates stay 0
            sample: Sample from the policy (RL rollouts) instead of acting deterministically
            threshold: Gate probability threshold for deterministic routing

        Returns:
            gate_actions: (batch, 4) float {0, 1}
            expert_raw: (batch, n_moe_layers * n_experts) raw bias action (see expert_bias())
        """
        probs = torch.sigmoid(decisions['gate_logits'])
        mean = decisions['expert_raw']
        if sample:
            gate_actions = torch.bernoulli(probs)
            expert_raw = Normal(mean, self.expert_log_std.exp()).sample()
        else:
            gate_actions = (probs > threshold).float()
            expert_raw = mean
        return gate_actions * gate_mask, expert_raw

    def log_prob(
        self,
        decisions: Dict[str, torch.Tensor],
        gate_actions: torch.Tensor,
        expert_raw: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability (batch,) of the actions that affected the outcome.

        `action_mask` is (batch, 5): one flag per gate plus the expert-bias
        flag. A masked gate (missing component, or ignored because the bypass
        path was taken) contributes nothing, and so does the expert bias when
        it was not applied.
        """
        gate_lp = Bernoulli(logits=decisions['gate_logits']).log_prob(gate_actions)
        log_prob = (gate_lp * action_mask[:, :EXPERT]).sum(dim=-1)
        dist = Normal(decisions['expert_raw'], self.expert_log_std.exp())
        return log_prob + dist.log_prob(expert_raw).sum(dim=-1) * action_mask[:, EXPERT]


class StateSummaryEncoder(nn.Module):
    """
    Encodes current state into a fixed-size summary for meta-controller input.

    State features are query-side signals, not answer correctness estimates:
    - Query predictability: mean next-token entropy over the query
    - Context length indicator
    - Memory fill / availability
    - Mean top-1 next-token probability over the query
    """

    def __init__(self, state_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim

        # Learnable projection for various state features
        self.uncertainty_proj = nn.Linear(1, 32)
        self.context_proj = nn.Linear(1, 32)
        self.memory_proj = nn.Linear(2, 32)  # episodic + semantic flags
        self.confidence_proj = nn.Linear(1, 32)

        self.output_proj = nn.Linear(128, state_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        uncertainty: torch.Tensor,  # (batch, 1)
        context_length: torch.Tensor,  # (batch, 1) normalized
        memory_usage: torch.Tensor,  # (batch, 2) [episodic, semantic]
        confidence: torch.Tensor  # (batch, 1)
    ) -> torch.Tensor:
        """
        Encode state features into summary vector.

        Returns:
            (batch, state_dim) state summary
        """
        u = self.activation(self.uncertainty_proj(uncertainty))
        c = self.activation(self.context_proj(context_length))
        m = self.activation(self.memory_proj(memory_usage))
        conf = self.activation(self.confidence_proj(confidence))

        # Concatenate and project
        combined = torch.cat([u, c, m, conf], dim=-1)
        state = self.output_proj(combined)

        return state
