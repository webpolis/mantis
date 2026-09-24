"""
Meta-Controller for MANTIS Architecture

Residual MLP policy that outputs routing decisions for dynamic query processing.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from typing import Dict, Tuple

GATES = ('early_exit', 'episodic', 'semantic', 'verification')


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

    The meta-controller analyzes query complexity and outputs 5 routing gates:
    - Early Exit: Skip deep processing for simple queries
    - Episodic Access: Query recent interaction history
    - Semantic Retrieval: Access long-term knowledge base
    - Expert Selection: Additive bias on the MoE gate logits
    - Verification: Trigger critic model for fact-checking

    The input is one vector per query, so the network is a stack of residual
    MLP blocks. As a policy, gates are Bernoulli and the expert bias is a
    diagonal Gaussian around `expert_weights`; deterministic routing uses the
    gate probabilities against a threshold and the Gaussian mean.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 6,
        d_ff: int = 4096,
        dropout: float = 0.1,
        n_experts: int = 8,
        state_dim: int = 128
    ):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.state_dim = state_dim

        self.embedding = nn.Linear(d_model + state_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualMLPBlock(d_model, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.gate_head = nn.Linear(d_model, len(GATES))
        self.expert_selector = nn.Linear(d_model, n_experts)
        self.expert_log_std = nn.Parameter(torch.full((n_experts,), math.log(0.5)))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
                - early_exit / episodic / semantic / verification: (batch, 1) probabilities
                - gate_logits: (batch, 4) logits of those gates, in GATES order
                - expert_weights: (batch, n_experts) raw logits used as additive gate bias
        """
        x = self.embedding(torch.cat([query_embedding, state_summary], dim=-1))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        h = self.final_norm(x)

        gate_logits = self.gate_head(h)
        decisions = {gate: torch.sigmoid(gate_logits[:, i:i + 1]) for i, gate in enumerate(GATES)}
        decisions['gate_logits'] = gate_logits
        decisions['expert_weights'] = self.expert_selector(h)
        return decisions

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
            expert_bias: (batch, n_experts) bias applied to the MoE gate logits
        """
        probs = torch.sigmoid(decisions['gate_logits'])
        mean = decisions['expert_weights']
        if sample:
            gate_actions = torch.bernoulli(probs)
            expert_bias = Normal(mean, self.expert_log_std.exp()).sample()
        else:
            gate_actions = (probs > threshold).float()
            expert_bias = mean
        return gate_actions * gate_mask, expert_bias

    def log_prob(
        self,
        decisions: Dict[str, torch.Tensor],
        gate_actions: torch.Tensor,
        expert_bias: torch.Tensor,
        gate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability (batch,) of taken actions. Masked gates contribute
        nothing; the expert bias only counts when the base model has experts.
        """
        gate_lp = Bernoulli(logits=decisions['gate_logits']).log_prob(gate_actions)
        log_prob = (gate_lp * gate_mask).sum(dim=-1)
        if self.n_experts > 1:
            dist = Normal(decisions['expert_weights'], self.expert_log_std.exp())
            log_prob = log_prob + dist.log_prob(expert_bias).sum(dim=-1)
        return log_prob


class StateSummaryEncoder(nn.Module):
    """
    Encodes current state into a fixed-size summary for meta-controller input.

    State includes:
    - Previous uncertainty estimates
    - Context length indicator
    - Recent memory access patterns
    - Average confidence scores
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
