"""
Episodic Memory System

SSM-based compression of recent interactions into continuous state vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import deque
from ..models.ssm import EpisodicMemorySSM


class EpisodicMemory:
    """
    Manages recent interaction history using SSM compression.

    Stores:
    - Raw token sequences (buffer)
    - Compressed state vectors
    - Importance scores for consolidation
    """

    def __init__(
        self,
        ssm_model: EpisodicMemorySSM,
        max_entries: int = 100,
        context_window: int = 8192,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.ssm = ssm_model.to(device)
        self.device = device
        self.max_entries = max_entries
        self.context_window = context_window

        # Memory buffers
        self.entries = deque(maxlen=max_entries)
        self.importance_scores = deque(maxlen=max_entries)

    def add(
        self,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add new interaction to episodic memory.

        Args:
            tokens: (seq_len,) token ids
            embeddings: (seq_len, d_model) embedded sequence
            metadata: Optional metadata (timestamp, user_id, etc.)
        """
        # Compress to state vector
        with torch.no_grad():
            embeddings = embeddings.unsqueeze(0).to(self.device)  # (1, seq_len, d_model)
            state = self.ssm.encode_sequence(embeddings)  # (1, d_state)
            state = state.squeeze(0).cpu()  # (d_state,)

        # Compute importance score (using embedding norm as proxy)
        importance = embeddings.norm(dim=-1).mean().item()

        # Store entry
        entry = {
            'tokens': tokens.cpu(),
            'embeddings': embeddings.squeeze(0).cpu(),
            'state': state,
            'metadata': metadata or {},
            'timestamp': torch.tensor(len(self.entries), dtype=torch.float32)
        }

        self.entries.append(entry)
        self.importance_scores.append(importance)

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 3,
        return_tokens: bool = False
    ) -> List[str]:
        """
        Retrieve most relevant recent memories.

        Args:
            query_embedding: (d_model,) query vector
            top_k: Number of memories to retrieve
            return_tokens: Return token ids instead of text

        Returns:
            List of retrieved memory texts or token sequences
        """
        if len(self.entries) == 0:
            return []

        query_embedding = query_embedding.to(self.device)

        # Project query_embedding to d_state space if dimensions don't match
        if query_embedding.size(-1) != self.ssm.d_state:
            with torch.no_grad():
                if query_embedding.dim() == 1:
                    query_embedding = query_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
                elif query_embedding.dim() == 2:
                    query_embedding = query_embedding.unsqueeze(0)  # (1, seq_len, d_model)
                # Always use full SSM + mean pool + projection for consistent embeddings
                _, query_state = self.ssm(query_embedding, return_state=True)
                query_embedding = query_state.squeeze(0)  # (d_state,)

        # Compute similarities with stored states
        scores = []
        for entry in self.entries:
            state = entry['state'].to(self.device)

            # Cosine similarity
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                state.unsqueeze(0),
                dim=-1
            ).item()

            scores.append(similarity)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Return retrieved memories
        retrieved = []
        for idx in top_indices:
            entry = self.entries[idx]
            if return_tokens:
                retrieved.append(entry['tokens'])
            else:
                # Convert tokens to text (placeholder - needs tokenizer)
                retrieved.append(f"<memory_{idx}>")

        return retrieved

    def get_high_importance(self, threshold: float = 0.7) -> List[Dict]:
        """
        Get high-importance memories for consolidation.

        Args:
            threshold: Importance threshold (0-1)

        Returns:
            List of high-importance entries
        """
        if len(self.entries) == 0:
            return []

        # Normalize importance scores
        scores = torch.tensor(list(self.importance_scores))
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Filter by threshold
        candidates = []
        for i, score in enumerate(scores):
            if score >= threshold:
                candidates.append(self.entries[i])

        return candidates

    def remove(self, entries_to_remove: List[Dict]) -> None:
        """
        Remove consolidated memories from episodic buffer.
        """
        # Simple removal by timestamp matching
        timestamps_to_remove = {e['timestamp'].item() for e in entries_to_remove}

        new_entries = deque()
        new_scores = deque()

        for entry, score in zip(self.entries, self.importance_scores):
            if entry['timestamp'].item() not in timestamps_to_remove:
                new_entries.append(entry)
                new_scores.append(score)

        self.entries = new_entries
        self.importance_scores = new_scores

    def size(self) -> int:
        """Return number of stored memories."""
        return len(self.entries)

    def clear(self) -> None:
        """Clear all memories."""
        self.entries.clear()
        self.importance_scores.clear()

    def get_state_summary(self) -> torch.Tensor:
        """
        Get summary of episodic memory state.

        Returns:
            (d_state,) aggregated state vector
        """
        if len(self.entries) == 0:
            return torch.zeros(self.ssm.d_state)

        # Aggregate states (weighted by recency)
        states = torch.stack([e['state'] for e in self.entries])

        # Exponential decay weights (recent = higher)
        weights = torch.exp(-0.1 * torch.arange(len(states), 0, -1))
        weights = weights / weights.sum()

        # Weighted sum
        aggregated = (states * weights.unsqueeze(-1)).sum(dim=0)

        return aggregated


def group_similar(entries: List[Dict], threshold: float = 0.8) -> List[List[Dict]]:
    """
    Group similar memory entries for consolidation.

    Args:
        entries: List of memory entries
        threshold: Similarity threshold for grouping

    Returns:
        List of groups (each group is a list of similar entries)
    """
    if not entries:
        return []

    # Extract state vectors
    states = torch.stack([e['state'] for e in entries])

    # Compute pairwise similarities
    similarities = F.cosine_similarity(
        states.unsqueeze(1),
        states.unsqueeze(0),
        dim=-1
    )

    # Greedy clustering
    used = set()
    groups = []

    for i in range(len(entries)):
        if i in used:
            continue

        # Start new group
        group = [entries[i]]
        used.add(i)

        # Add similar entries
        for j in range(i + 1, len(entries)):
            if j not in used and similarities[i, j] >= threshold:
                group.append(entries[j])
                used.add(j)

        groups.append(group)

    return groups
