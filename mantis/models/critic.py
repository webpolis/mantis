"""
Critic Model for Hallucination Detection and Verification

A small transformer encoder over the frozen base model's final hidden states
of [evidence; query; response]. Reading the backbone's representations gives
the critic the backbone's language knowledge instead of asking a randomly
initialized encoder to learn language from a small labeled set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple

SEGMENTS = ('evidence', 'query', 'response')

# Share of the token budget each part gets first; leftover capacity then goes
# to the response, the evidence and the query, in that order.
BUDGET_SHARES = {'response': 0.5, 'evidence': 0.35, 'query': 0.15}


class CriticModel(nn.Module):
    """
    Verification model: P(response is correct | evidence, query).

    Input order is [evidence; query; response] so that, under the causal
    backbone, every response token's hidden state has seen the evidence and
    the query. Segment embeddings mark the three parts. One sigmoid head
    scores correctness; `temperature` (fitted on a calibration split in
    Stage 4) rescales its logit so the score is a calibrated probability.
    """

    def __init__(
        self,
        d_input: int = 2048,
        d_model: int = 1024,
        n_layers: int = 6,
        n_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(d_input, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(len(SEGMENTS), d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.register_buffer('temperature', torch.ones(()))

        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, std=0.02)

    @classmethod
    def from_config(cls, critic_config, base_config) -> 'CriticModel':
        return cls(d_input=base_config.d_model, **vars(critic_config))

    def forward(
        self,
        hidden: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, d_input) base-model final hidden states
            segment_ids: (batch, seq_len) 0=evidence, 1=query, 2=response
            attention_mask: Optional (batch, seq_len), 1 = real token

        Returns:
            logits: (batch, 1) correctness logits (before temperature)
        """
        batch_size, seq_len = segment_ids.shape
        positions = torch.arange(seq_len, device=segment_ids.device).unsqueeze(0)
        x = self.input_proj(hidden.float()) + self.position_embedding(positions) + self.segment_embedding(segment_ids)
        x = self.dropout(x)

        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        if attention_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

    def probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated correctness probability from forward() logits."""
        return torch.sigmoid(logits / self.temperature)

    def build_input(
        self,
        query_ids: Sequence[int],
        response_ids: Sequence[int],
        evidence_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Concatenate [evidence, query, response] within max_seq_len.

        Each part first receives its BUDGET_SHARES share of the budget (the
        query keeps its last tokens, the others their first); leftover
        capacity then extends the response, the evidence and the query, in
        that order, so a short evidence block never starves the response.

        Returns:
            (input_ids, segment_ids) as lists
        """
        parts = {'evidence': list(evidence_ids or []), 'query': list(query_ids), 'response': list(response_ids)}
        budget = self.max_seq_len
        take = {name: min(len(ids), int(BUDGET_SHARES[name] * budget)) for name, ids in parts.items()}
        leftover = budget - sum(take.values())
        for name in ('response', 'evidence', 'query'):
            extra = min(len(parts[name]) - take[name], leftover)
            take[name] += extra
            leftover -= extra

        evidence = parts['evidence'][:take['evidence']]
        query = parts['query'][len(parts['query']) - take['query']:] if take['query'] else []
        response = parts['response'][:take['response']]
        input_ids = evidence + query + response
        segment_ids = [0] * len(evidence) + [1] * len(query) + [2] * len(response)
        return input_ids, segment_ids

    @staticmethod
    def collate(rows: List[Tuple[List[int], List[int]]], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad (input_ids, segment_ids) rows into batch tensors plus an attention mask."""
        length = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
        segment_ids = torch.zeros(len(rows), length, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), length, dtype=torch.long)
        for i, (ids, segs) in enumerate(rows):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            segment_ids[i, :len(segs)] = torch.tensor(segs)
            attention_mask[i, :len(ids)] = 1
        return input_ids.to(device), segment_ids.to(device), attention_mask.to(device)

    @staticmethod
    @torch.no_grad()
    def backbone_features(base_model, input_ids: torch.Tensor) -> torch.Tensor:
        """Frozen base-model final hidden states, (batch, seq_len, d_input) in float32."""
        return base_model(input_ids, return_hidden=True)['last_hidden'].float()

    @torch.no_grad()
    def score(self, base_model, input_ids: Sequence[int], segment_ids: Sequence[int]) -> float:
        """Calibrated correctness probability of one build_input() row."""
        device = self.input_proj.weight.device
        hidden = self.backbone_features(base_model, torch.tensor([list(input_ids)], device=device))
        logits = self.forward(hidden, torch.tensor([list(segment_ids)], device=device))
        return self.probability(logits).item()

    @torch.no_grad()
    def verify(
        self,
        base_model,
        query_ids: Sequence[int],
        response_ids: Sequence[int],
        evidence_ids: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Score one (query, response, evidence) triple given as token ID lists.

        Returns:
            Calibrated correctness probability
        """
        return self.score(base_model, *self.build_input(query_ids, response_ids, evidence_ids))

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy on correctness logits; targets are (batch, 1) in {0, 1}."""
        return F.binary_cross_entropy_with_logits(logits, targets)

    @torch.no_grad()
    def fit_temperature(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Temperature scaling on a held-out calibration split: pick the scalar
        that minimizes the NLL of sigmoid(logits / T). Stores and returns T.
        """
        candidates = torch.exp(torch.linspace(-2.5, 2.5, 201, device=logits.device))
        nll = torch.stack([
            F.binary_cross_entropy_with_logits(logits / t, targets) for t in candidates
        ])
        best = candidates[int(nll.argmin())]
        self.temperature.copy_(best)
        return float(best)


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
