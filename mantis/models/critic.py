"""
Critic Model for Hallucination Detection and Verification

Small transformer encoder with binary classification for fact-checking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CriticModel(nn.Module):
    """
    Verification model for hallucination detection.

    Takes (query, response, retrieved_facts) and outputs:
    - Correctness probability
    - Confidence score
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(3, d_model)  # query, response, facts
        self.dropout = nn.Dropout(dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Classification heads
        self.correctness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for verification.

        Args:
            input_ids: (batch, seq_len) concatenated [query, response, facts]
            segment_ids: (batch, seq_len) 0=query, 1=response, 2=facts
            attention_mask: Optional (batch, seq_len)

        Returns:
            correctness: (batch, 1) probability in [0, 1]
            confidence: (batch, 1) confidence score in [0, 1]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = (
            self.token_embedding(input_ids) +
            self.position_embedding(positions) +
            self.segment_embedding(segment_ids)
        )
        x = self.dropout(x)

        # TransformerEncoder expects True for padded positions to IGNORE
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over real tokens (no dedicated CLS token in this architecture)
        if attention_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # Predictions
        correctness = self.correctness_head(pooled)
        confidence = self.confidence_head(pooled)

        return correctness, confidence

    def build_input(
        self,
        query_ids: list,
        response_ids: list,
        facts_ids: Optional[list] = None,
    ) -> Tuple[list, list]:
        """
        Concatenate [query, response, facts] within max_seq_len.

        Query keeps its last max_seq_len/4 tokens, facts their first
        max_seq_len/4, and the response gets the rest (truncated at the end).

        Returns:
            (input_ids, segment_ids) as lists
        """
        quarter = self.max_seq_len // 4
        query = list(query_ids)[-quarter:]
        facts = list(facts_ids or [])[:quarter]
        response = list(response_ids)[:self.max_seq_len - len(query) - len(facts)]
        input_ids = query + response + facts
        segment_ids = [0] * len(query) + [1] * len(response) + [2] * len(facts)
        return input_ids, segment_ids

    @torch.no_grad()
    def verify(
        self,
        query_ids: list,
        response_ids: list,
        facts_ids: Optional[list] = None
    ) -> float:
        """
        Score one (query, response, facts) triple given as token ID lists.

        Returns:
            Scalar correctness probability
        """
        input_ids, segment_ids = self.build_input(query_ids, response_ids, facts_ids)
        device = self.token_embedding.weight.device
        correctness, _ = self.forward(
            torch.tensor([input_ids], device=device),
            torch.tensor([segment_ids], device=device),
        )
        return correctness.item()

    def compute_loss(
        self,
        correctness_pred: torch.Tensor,
        confidence_pred: torch.Tensor,
        correctness_target: torch.Tensor,
        confidence_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            correctness_pred: (batch, 1) predicted correctness
            confidence_pred: (batch, 1) predicted confidence
            correctness_target: (batch, 1) ground truth {0, 1}
            confidence_target: Optional (batch, 1) ground truth confidence

        Returns:
            Total loss
        """
        # Binary cross-entropy for correctness
        correctness_loss = F.binary_cross_entropy(
            correctness_pred,
            correctness_target
        )

        # Optional confidence loss
        if confidence_target is not None:
            confidence_loss = F.mse_loss(confidence_pred, confidence_target)
        else:
            # Encourage high confidence for correct, low for incorrect
            confidence_target = correctness_target
            confidence_loss = F.mse_loss(confidence_pred, confidence_target)

        total_loss = correctness_loss + 0.5 * confidence_loss

        return total_loss


class CriticValueNetwork(nn.Module):
    """
    Value network for PPO training of meta-controller.

    Estimates expected reward for a given state.
    """

    def __init__(
        self,
        d_model: int = 1024,
        state_dim: int = 128,
        n_layers: int = 3,
        hidden_dim: int = 512
    ):
        super().__init__()

        input_dim = d_model + state_dim

        layers = []
        prev_dim = input_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of state.

        Args:
            state: (batch, d_model + state_dim)

        Returns:
            value: (batch, 1)
        """
        return self.network(state)
