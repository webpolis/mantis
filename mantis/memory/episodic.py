"""
Episodic Memory System

SSM-based compression of recent interactions into continuous state vectors.
"""

import threading
import time

import torch
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional
from collections import deque
from ..models.ssm import EpisodicMemorySSM


class EpisodicMemory:
    """
    Manages recent interaction history using SSM compression.

    Each entry stores its token IDs, the base-model hidden states it was built
    from, the compressed SSM state and an importance score. Entries carry a
    unique, monotonically increasing `id`. All reads and writes of the buffer
    hold a lock; readers work on snapshots.
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
        self._lock = threading.Lock()
        self._next_id = 0
        self.entries: deque = deque(maxlen=max_entries)

    @torch.no_grad()
    def _encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(seq_len, d_model) or (d_model,) -> (d_state,) SSM state."""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return self.ssm.encode_sequence(embeddings.unsqueeze(0).to(self.device)).squeeze(0).cpu()

    def add(
        self,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a new interaction to episodic memory.

        Args:
            tokens: (seq_len,) token ids
            embeddings: (seq_len, d_model) base-model hidden states for `tokens`
            metadata: Optional metadata (user_id, source, ...)

        Returns:
            The entry id
        """
        tokens = tokens[-self.context_window:].cpu()
        embeddings = embeddings[-self.context_window:].detach().cpu()
        state = self._encode(embeddings)

        entry = {
            'tokens': tokens,
            'embeddings': embeddings,
            'state': state,
            # Mean hidden-state norm as an importance proxy
            'importance': embeddings.norm(dim=-1).mean().item(),
            'metadata': metadata or {},
            'created': time.time(),
        }
        with self._lock:
            entry['id'] = self._next_id
            self._next_id += 1
            self.entries.append(entry)
        return entry['id']

    def snapshot(self) -> List[Dict]:
        """Current entries, oldest first."""
        with self._lock:
            return list(self.entries)

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> List[torch.Tensor]:
        """
        Retrieve the token IDs of the most similar stored interactions.

        Args:
            query_embedding: (d_model,) pooled query or (seq_len, d_model) hidden states
            top_k: Number of memories to retrieve

        Returns:
            List of (seq_len,) token tensors, most similar first
        """
        entries = self.snapshot()
        if not entries:
            return []

        query_state = self._encode(query_embedding.detach())
        states = torch.stack([e['state'] for e in entries])
        scores = F.cosine_similarity(states, query_state.unsqueeze(0), dim=-1)
        top = torch.topk(scores, min(top_k, len(entries))).indices.tolist()
        return [entries[i]['tokens'] for i in top]

    def get_high_importance(self, threshold: float = 0.7) -> List[Dict]:
        """
        Get high-importance memories for consolidation.

        Args:
            threshold: Importance threshold (0-1) after min-max normalization

        Returns:
            List of high-importance entries (snapshot)
        """
        entries = self.snapshot()
        if not entries:
            return []

        scores = torch.tensor([e['importance'] for e in entries])
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return [e for e, s in zip(entries, scores.tolist()) if s >= threshold]

    def remove(self, entries_to_remove: Iterable[Dict]) -> None:
        """Remove entries (matched by id) from the buffer."""
        ids = {e['id'] for e in entries_to_remove}
        with self._lock:
            self.entries = deque((e for e in self.entries if e['id'] not in ids), maxlen=self.max_entries)

    def size(self) -> int:
        """Return number of stored memories."""
        with self._lock:
            return len(self.entries)

    def clear(self) -> None:
        """Clear all memories."""
        with self._lock:
            self.entries.clear()

    def get_state_summary(self) -> torch.Tensor:
        """
        Get summary of episodic memory state.

        Returns:
            (d_state,) aggregated state vector, weighted toward recent entries
        """
        entries = self.snapshot()
        if not entries:
            return torch.zeros(self.ssm.d_state)

        states = torch.stack([e['state'] for e in entries])
        weights = torch.exp(-0.1 * torch.arange(len(states), 0, -1))
        weights = weights / weights.sum()
        return (states * weights.unsqueeze(-1)).sum(dim=0)


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

    states = torch.stack([e['state'] for e in entries])
    similarities = F.cosine_similarity(states.unsqueeze(1), states.unsqueeze(0), dim=-1)

    # Greedy clustering
    used = set()
    groups = []
    for i in range(len(entries)):
        if i in used:
            continue
        group = [entries[i]]
        used.add(i)
        for j in range(i + 1, len(entries)):
            if j not in used and similarities[i, j] >= threshold:
                group.append(entries[j])
                used.add(j)
        groups.append(group)

    return groups
