"""
Episodic Memory System

Recent interactions, each encoded by the SSM into a compact retrieval key.
"""

import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..models.ssm import EpisodicMemorySSM
from .provenance import trust_level

Segments = Sequence[Tuple[str, torch.Tensor]]


class EpisodicMemory:
    """
    Bounded buffer of recent interactions.

    Each entry keeps its token IDs (as role-labelled segments), a compact SSM
    state used as the retrieval key, the pooled base-model embedding used for
    reranking against semantic evidence, a retrieval hit count, provenance
    metadata (namespace, source, trust, timestamp) and a unique id. Full
    hidden-state tensors are not retained.

    The buffer holds at most `max_entries`. Adding beyond that evicts the
    oldest entry, which is handed to `on_overflow` first, so the consolidator
    can write it to semantic memory before it disappears. All reads and
    writes hold a lock; readers work on snapshots.
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
        self.on_overflow: Optional[Callable[[List[Dict]], None]] = None
        self._add_lock = threading.Lock()
        self._lock = threading.Lock()
        self._next_id = 0
        self.entries: deque = deque()
        self.stats = {'writes': 0, 'evictions': 0}

    @torch.no_grad()
    def _encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """(seq_len, d_model) or (d_model,) -> (d_state,) SSM state."""
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        return self.ssm.encode_sequence(hidden.unsqueeze(0).to(self.device).float()).squeeze(0).cpu()

    def add(
        self,
        hidden: torch.Tensor,
        segments: Segments,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Add an interaction.

        Args:
            hidden: (seq_len, d_model) base-model final hidden states of the
                concatenated segment tokens (the last `context_window` are used)
            segments: (role, token_ids) pairs, e.g. [('query', ...), ('response', ...)]
            metadata: provenance: namespace, source, trust, verified, ...

        Returns:
            The entry id
        """
        hidden = hidden[-self.context_window:].detach().float().cpu()
        tokens = torch.cat([torch.as_tensor(ids).long().cpu() for _, ids in segments])
        metadata = {'namespace': 'default', 'source': 'interaction', 'timestamp': time.time(), **(metadata or {})}
        metadata['trust'] = trust_level(metadata)
        entry = {
            'segments': [(role, torch.as_tensor(ids).long().cpu()) for role, ids in segments],
            'tokens': tokens[-self.context_window:],
            'state': self._encode(hidden),
            'embedding': hidden.mean(dim=0),
            'hits': 0,
            'metadata': metadata,
            'created': time.time(),
        }
        with self._add_lock:
            with self._lock:
                evicted = list(self.entries)[:max(0, len(self.entries) + 1 - self.max_entries)]
            # Preserve the old entries before removing their only copy. The
            # hook writes outside the buffer lock so persistence can snapshot it.
            if evicted and self.on_overflow is not None:
                self.on_overflow(evicted)
            with self._lock:
                entry['id'] = self._next_id
                self._next_id += 1
                for _ in evicted:
                    self.entries.popleft()
                self.entries.append(entry)
                self.stats['writes'] += 1
                self.stats['evictions'] += len(evicted)
        return entry['id']

    def snapshot(self) -> List[Dict]:
        """Current entries, oldest first."""
        with self._lock:
            return list(self.entries)

    def retrieve(
        self,
        query_hidden: torch.Tensor,
        top_k: int = 3,
        namespace: Optional[str] = None,
        exclude_ids: Iterable[int] = (),
    ) -> List[Dict]:
        """
        Most similar stored interactions (by SSM-state cosine), most similar first.

        Args:
            query_hidden: (seq_len, d_model) query hidden states or a (d_model,) pooled vector
            top_k: Number of entries to return
            namespace: Only entries of this namespace (None = all)
            exclude_ids: Entry ids to skip (e.g. evidence already used)

        Returns:
            Entry dicts (snapshots) with an extra 'score'; each hit increments the entry's hit count
        """
        excluded = set(exclude_ids)
        entries = [e for e in self.snapshot()
                   if e['id'] not in excluded and (namespace is None or e['metadata'].get('namespace') == namespace)]
        if not entries:
            return []

        query_state = self._encode(query_hidden.detach())
        states = torch.stack([e['state'] for e in entries])
        scores = F.cosine_similarity(states, query_state.unsqueeze(0), dim=-1)
        top = torch.topk(scores, min(top_k, len(entries))).indices.tolist()
        hits = []
        with self._lock:
            for i in top:
                entries[i]['hits'] += 1
                hits.append({**entries[i], 'score': float(scores[i])})
        return hits

    def candidates(self, min_hits: int = 1) -> List[Dict]:
        """Entries retrieved at least `min_hits` times, or marked important: consolidation candidates."""
        return [e for e in self.snapshot() if e['hits'] >= min_hits or e['metadata'].get('important')]

    def remove(self, entries_to_remove: Iterable[Dict]) -> None:
        """Remove entries (matched by id) from the buffer."""
        ids = {e['id'] for e in entries_to_remove}
        with self._add_lock:
            with self._lock:
                self.entries = deque(e for e in self.entries if e['id'] not in ids)

    def size(self) -> int:
        """Return number of stored memories."""
        with self._lock:
            return len(self.entries)

    def clear(self) -> None:
        """Clear all memories."""
        with self._add_lock:
            with self._lock:
                self.entries.clear()

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Save the buffer (entries and id counter) to a torch file."""
        with self._lock:
            torch.save({'entries': list(self.entries), 'next_id': self._next_id, 'stats': dict(self.stats)}, path)

    def load(self, path: str) -> int:
        """Restore a buffer saved with save(); returns the number of entries."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        with self._add_lock:
            with self._lock:
                self.entries = deque(data['entries'])
                self._next_id = max(data['next_id'], self._next_id)
                self.stats.update(data.get('stats', {}))
                return len(self.entries)
