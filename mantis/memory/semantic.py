"""
Semantic Memory System

Long-term knowledge storage using a FAISS vector index.
"""

import math
import os
import pickle
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn


class SemanticMemory:
    """
    Long-term knowledge storage using a FAISS vector index.

    - Every vector has a stable ID; metadata lives in an ordered dict keyed by it.
    - Eviction (FIFO) tombstones IDs instead of shifting positions. Searches
      over-fetch by the tombstone count and skip evicted IDs; the index is
      rebuilt once more than 20% of its vectors are tombstones.
    - IVF indexes serve exact flat search until MIN_IVF_TRAIN entries exist,
      then train with a cluster count sized to the data.
    - An optional `projection` (trained in Stage 2) maps base-model embeddings
      into retrieval space. Vectors are L2-normalized before indexing.
    - All operations hold a re-entrant lock, so background consolidation and
      inference can share one instance.
    """

    MIN_IVF_TRAIN = 10_000

    def __init__(
        self,
        dimension: int = 2048,
        max_entries: int = 1_000_000,
        index_type: str = 'IVF',
        use_gpu: bool = True,
        projection: Optional[nn.Module] = None,
    ):
        if index_type not in ('Flat', 'IVF', 'HNSW'):
            raise ValueError(f"Unknown index type: {index_type}")
        self.dimension = dimension
        self.max_entries = max_entries
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.projection = projection

        self._lock = threading.RLock()
        self.entries: 'OrderedDict[int, Dict]' = OrderedDict()
        self._next_id = 0
        self._stale = 0
        self.index_trained = index_type != 'IVF'
        self._quantizer = None
        self._gpu_resources = None
        self.index = self._to_device(self._empty_index())

    # ------------------------------------------------------------------ index

    def _empty_index(self):
        if self.index_type == 'HNSW':
            base = faiss.IndexHNSWFlat(self.dimension, 32)
            base.hnsw.efConstruction = 40
            base.hnsw.efSearch = 16
        else:
            base = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIDMap(base)

    def _ivf_index(self, vectors: np.ndarray, ids: np.ndarray):
        """Train an IVF-PQ index sized to `vectors` (≥39 points per cluster)."""
        n = len(vectors)
        nlist = max(1, min(int(4 * math.sqrt(n)), n // 39))
        n_subquantizers = max(m for m in range(1, min(64, self.dimension) + 1) if self.dimension % m == 0)
        self._quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(self._quantizer, self.dimension, nlist, n_subquantizers, 8)
        index.train(vectors)
        index.nprobe = min(nlist, 16)
        index.add_with_ids(vectors, ids)
        return index

    def _to_device(self, index):
        if not self.use_gpu:
            return index
        try:
            self._gpu_resources = self._gpu_resources or faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
        except Exception as e:
            print(f"Keeping FAISS index on CPU: {e}")
            return index

    def _rebuild(self):
        """Rebuild the index from live entries, dropping tombstones."""
        ids = np.fromiter(self.entries.keys(), dtype=np.int64, count=len(self.entries))
        vectors = (np.vstack([e['vector'] for e in self.entries.values()])
                   if self.entries else np.zeros((0, self.dimension), dtype='float32'))

        if self.index_type == 'IVF' and len(ids) >= self.MIN_IVF_TRAIN:
            print(f"Training FAISS IVF index on {len(ids)} entries...")
            index = self._ivf_index(vectors, ids)
            self.index_trained = True
        else:
            index = self._empty_index()
            if len(ids):
                index.add_with_ids(vectors, ids)
            self.index_trained = self.index_type != 'IVF'
        self.index = self._to_device(index)
        self._stale = 0

    # ------------------------------------------------------------ vectorizing

    def _vectors(self, embeddings) -> np.ndarray:
        """(n, dimension) float32, projected and L2-normalized."""
        x = torch.as_tensor(embeddings).detach().float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.projection is not None:
            with torch.no_grad():
                device = next(self.projection.parameters()).device
                x = self.projection(x.to(device)).float()
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return np.ascontiguousarray(x.cpu().numpy(), dtype='float32')

    # ------------------------------------------------------------------ write

    def add(self, embedding, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a new entry to semantic memory.

        Args:
            embedding: (dimension,) base-model embedding (tensor or array)
            text: Original text of the fact
            metadata: Optional dict with source, confidence, timestamp, etc.

        Returns:
            The entry id
        """
        return self.add_batch(embedding, [text], [metadata or {}])[0]

    def add_batch(self, embeddings, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[int]:
        """
        Add multiple entries at once.

        Args:
            embeddings: (n, dimension) batch of embeddings
            texts: List of text strings
            metadata_list: Optional list of metadata dicts

        Returns:
            List of entry ids
        """
        vectors = self._vectors(embeddings)
        n = len(vectors)
        if len(texts) != n:
            raise ValueError(f"{n} embeddings but {len(texts)} texts")
        metadata_list = metadata_list or [{} for _ in range(n)]

        with self._lock:
            overflow = len(self.entries) + n - self.max_entries
            if overflow > 0:
                self._evict(max(overflow, int(0.1 * self.max_entries)))

            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n
            self.index.add_with_ids(vectors, ids)
            for i, entry_id in enumerate(ids.tolist()):
                self.entries[entry_id] = {
                    'text': texts[i],
                    'vector': vectors[i:i + 1],
                    'metadata': metadata_list[i],
                }

            if (not self.index_trained and len(self.entries) >= self.MIN_IVF_TRAIN) \
                    or self._stale > 0.2 * max(1, self.index.ntotal):
                self._rebuild()
            return ids.tolist()

    def _evict(self, n: int):
        """Tombstone the n oldest entries (FIFO)."""
        for _ in range(min(n, len(self.entries))):
            self.entries.popitem(last=False)
            self._stale += 1

    # ------------------------------------------------------------------- read

    def _search(self, query_embedding, top_k: int) -> List[Tuple[Dict, float]]:
        vector = self._vectors(query_embedding)
        with self._lock:
            if not self.entries:
                return []
            k = min(top_k + self._stale, self.index.ntotal)
            distances, ids = self.index.search(vector, k)
            hits = []
            for entry_id, dist in zip(ids[0].tolist(), distances[0].tolist()):
                entry = self.entries.get(entry_id)
                if entry is not None:
                    hits.append((entry, dist))
                if len(hits) == top_k:
                    break
            return hits

    def retrieve(self, query_embedding, top_k: int = 5, return_distances: bool = False):
        """
        Retrieve most relevant facts.

        Args:
            query_embedding: (dimension,) base-model query embedding
            top_k: Number of results to return
            return_distances: Also return L2 distances

        Returns:
            List of fact texts, or (texts, distances)
        """
        hits = self._search(query_embedding, top_k)
        texts = [entry['text'] for entry, _ in hits]
        if return_distances:
            return texts, [dist for _, dist in hits]
        return texts

    def retrieve_with_metadata(self, query_embedding, top_k: int = 5) -> List[Dict]:
        """
        Returns:
            List of dicts with 'text', 'distance', 'metadata'
        """
        return [
            {'text': entry['text'], 'distance': dist, 'metadata': entry['metadata']}
            for entry, dist in self._search(query_embedding, top_k)
        ]

    def size(self) -> int:
        """Return number of live entries."""
        with self._lock:
            return len(self.entries)

    # ------------------------------------------------------------ persistence

    def save(self, path: str):
        """Save to `{path}.index` (FAISS) and `{path}.meta` (entries and counters)."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with self._lock:
            index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
            faiss.write_index(index, f"{path}.index")
            with open(f"{path}.meta", 'wb') as f:
                pickle.dump({
                    'dimension': self.dimension,
                    'max_entries': self.max_entries,
                    'index_type': self.index_type,
                    'index_trained': self.index_trained,
                    'entries': self.entries,
                    'next_id': self._next_id,
                    'stale': self._stale,
                    'projection': self.projection.cpu() if self.projection is not None else None,
                }, f)
        print(f"Saved semantic memory to {path} ({len(self.entries)} entries)")

    @classmethod
    def load(cls, path: str, use_gpu: bool = True) -> 'SemanticMemory':
        """Load a memory saved with save()."""
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)

        memory = cls(
            dimension=data['dimension'],
            max_entries=data['max_entries'],
            index_type=data['index_type'],
            use_gpu=use_gpu,
            projection=data['projection'],
        )
        memory.entries = data['entries']
        memory._next_id = data['next_id']
        memory._stale = data['stale']
        memory.index_trained = data['index_trained']
        memory.index = memory._to_device(faiss.read_index(f"{path}.index"))
        print(f"Loaded semantic memory from {path} ({len(memory.entries)} entries)")
        return memory
