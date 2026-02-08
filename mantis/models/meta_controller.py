"""
Meta-Controller for MANTIS Architecture

Lightweight transformer that outputs routing decisions for dynamic query processing.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class MetaController(nn.Module):
    """
    Lightweight transformer that outputs routing decisions.

    The meta-controller analyzes query complexity and outputs 5 routing gates:
    - Early Exit: Skip deep processing for simple queries
    - Episodic Access: Query recent interaction history
    - Semantic Retrieval: Access long-term knowledge base
    - Expert Selection: Route to specialized MoE experts
    - Verification: Trigger critic model for fact-checking
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 6,
        n_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        n_experts: int = 8,
        state_dim: int = 128
    ):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.state_dim = state_dim

        # Input embedding: combines query + state summary
        self.embedding = nn.Linear(d_model + state_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output heads for routing decisions
        self.early_exit_gate = nn.Linear(d_model, 1)
        self.episodic_gate = nn.Linear(d_model, 1)
        self.semantic_gate = nn.Linear(d_model, 1)
        self.verification_gate = nn.Linear(d_model, 1)
        self.expert_selector = nn.Linear(d_model, n_experts)
        self.uncertainty_head = nn.Linear(d_model, 1)

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
        Forward pass to compute routing decisions.

        Args:
            query_embedding: (batch, d_model) - Encoded query
            state_summary: (batch, state_dim) - Confidence scores, context indicators

        Returns:
            Dict with continuous routing decisions (pre-threshold):
                - early_exit: (batch, 1) in [0,1]
                - episodic: (batch, 1) in [0,1]
                - semantic: (batch, 1) in [0,1]
                - verification: (batch, 1) in [0,1]
                - expert_weights: (batch, n_experts) probability distribution
                - uncertainty: (batch, 1) in [0,1]
        """
        batch_size = query_embedding.size(0)

        # Combine inputs
        x = torch.cat([query_embedding, state_summary], dim=-1)  # (batch, d_model + state_dim)
        x = self.embedding(x)  # (batch, d_model)
        x = self.dropout(x)
        x = x.unsqueeze(1)  # (batch, 1, d_model)

        # Process through transformer
        h = self.transformer(x)  # (batch, 1, d_model)
        h = h.squeeze(1)  # (batch, d_model)

        # Compute routing decisions
        decisions = {
            'early_exit': torch.sigmoid(self.early_exit_gate(h)),  # (batch, 1)
            'episodic': torch.sigmoid(self.episodic_gate(h)),
            'semantic': torch.sigmoid(self.semantic_gate(h)),
            'verification': torch.sigmoid(self.verification_gate(h)),
            'expert_weights': self.expert_selector(h),  # (batch, n_experts) raw logits as additive bias
            'uncertainty': torch.sigmoid(self.uncertainty_head(h))  # (batch, 1)
        }

        return decisions

    def decide(
        self,
        decisions: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> Dict:
        """
        Convert continuous outputs to binary decisions.

        Args:
            decisions: Output from forward()
            threshold: Binary decision threshold for gates

        Returns:
            Dict with binary decisions and probabilities
        """
        batch_size = decisions['early_exit'].size(0)

        # For single-item batch, return scalars; otherwise return lists
        if batch_size == 1:
            return {
                'early_exit': (decisions['early_exit'].item() > threshold),
                'episodic': (decisions['episodic'].item() > threshold),
                'semantic': (decisions['semantic'].item() > threshold),
                'verification': (decisions['verification'].item() > threshold),
                'expert_weights': decisions['expert_weights'],  # Keep 2D shape (1, n_experts)
                'uncertainty': decisions['uncertainty'].item()
            }
        else:
            return {
                'early_exit': (decisions['early_exit'] > threshold).squeeze(-1).tolist(),
                'episodic': (decisions['episodic'] > threshold).squeeze(-1).tolist(),
                'semantic': (decisions['semantic'] > threshold).squeeze(-1).tolist(),
                'verification': (decisions['verification'] > threshold).squeeze(-1).tolist(),
                'expert_weights': decisions['expert_weights'],
                'uncertainty': decisions['uncertainty'].squeeze(-1).tolist()
            }

    def compute_gumbel_softmax(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> torch.Tensor:
        """
        Differentiable approximation to discrete sampling for training.

        Args:
            logits: Raw scores
            temperature: Softmax temperature (lower = more discrete)
            hard: If True, use straight-through estimator

        Returns:
            Soft or hard gate values
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        soft_gate = torch.softmax(y / temperature, dim=-1)

        if hard:
            # Straight-through estimator
            indices = soft_gate.argmax(dim=-1, keepdim=True)
            hard_gate = torch.zeros_like(soft_gate).scatter_(-1, indices, 1.0)
            # Keep soft gradients
            return hard_gate - soft_gate.detach() + soft_gate

        return soft_gate


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
