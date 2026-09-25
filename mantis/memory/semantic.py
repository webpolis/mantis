"""
Semantic Memory System

Long-term knowledge storage using a FAISS vector index.
"""

import math
import os
import pickle
import threading
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn

from .provenance import trust_level


class SemanticMemory:
    """
    Long-term knowledge storage using a FAISS vector index.

    - Every vector has a stable ID; entries live in an ordered dict keyed by it
      with text, the stored vector and provenance metadata (namespace, source,
      trust, timestamp, superseded_by, ...).
    - Eviction (FIFO) and deletion tombstone IDs instead of shifting positions.
      Searches over-fetch by a bounded amount and skip tombstones, superseded
      entries, other namespaces and untrusted sources; the index is rebuilt
      off-thread, with an atomic swap, once more than 20% of it is stale.
    - IVF indexes serve exact flat search until MIN_IVF_TRAIN entries exist,
      then train with a cluster count sized to the data. `recall_at_k()`
      measures the approximate index against exact search.
    - An optional `projection` (trained in Stage 2) maps base-model embeddings
      into retrieval space. Vectors are L2-normalized before indexing.
    - `embedding_fingerprint` records which base model and tokenizer produced
      the vectors; the engine refuses stores built by a different one.
    - All operations hold a re-entrant lock, so background consolidation and
      inference can share one instance.
    """

    MIN_IVF_TRAIN = 10_000
    MAX_FETCH = 4096

    def __init__(
        self,
        dimension: int = 2048,
        max_entries: int = 1_000_000,
        index_type: str = 'IVF',
        use_gpu: bool = True,
        projection: Optional[nn.Module] = None,
        embedding_fingerprint: Optional[str] = None,
    ):
        if index_type not in ('Flat', 'IVF', 'HNSW'):
            raise ValueError(f"Unknown index type: {index_type}")
        self.dimension = dimension
        self.max_entries = max_entries
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.projection = projection
        self.embedding_fingerprint = embedding_fingerprint

        self._lock = threading.RLock()
        self.entries: 'OrderedDict[int, Dict]' = OrderedDict()
        self._next_id = 0
        self._stale = 0
        self._rebuild_thread: Optional[threading.Thread] = None
        self.index_trained = index_type != 'IVF'
        self._quantizer = None
        self._gpu_resources = None
        self.index = self._to_device(self._empty_index())
        self.stats = {'writes': 0, 'evictions': 0, 'deletions': 0, 'rebuilds': 0, 'searches': 0}

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

    def _build(self, vectors: np.ndarray, ids: np.ndarray):
        """A fresh index over `vectors`; trains IVF when there is enough data."""
        if self.index_type == 'IVF' and len(ids) >= self.MIN_IVF_TRAIN:
            index, trained = self._ivf_index(vectors, ids), True
        else:
            index = self._empty_index()
            if len(ids):
                index.add_with_ids(vectors, ids)
            trained = self.index_type != 'IVF'
        return index, trained

    def _needs_rebuild(self) -> bool:
        return ((not self.index_trained and len(self.entries) >= self.MIN_IVF_TRAIN)
                or self._stale > 0.2 * max(1, self.index.ntotal))

    def rebuild(self) -> None:
        """
        Rebuild the index from live entries, dropping tombstones.

        The build runs without the lock on a snapshot; entries added meanwhile
        are appended to the new index before the atomic swap, and entries
        evicted meanwhile become its initial tombstones.
        """
        with self._lock:
            snapshot_next_id = self._next_id
            ids = np.fromiter(self.entries.keys(), dtype=np.int64, count=len(self.entries))
            vectors = (np.vstack([e['vector'] for e in self.entries.values()])
                       if self.entries else np.zeros((0, self.dimension), dtype='float32'))

        index, trained = self._build(vectors, ids)

        with self._lock:
            newer = [(i, e) for i, e in self.entries.items() if i >= snapshot_next_id]
            if newer:
                index.add_with_ids(np.vstack([e['vector'] for _, e in newer]),
                                   np.fromiter((i for i, _ in newer), dtype=np.int64, count=len(newer)))
            self.index = self._to_device(index)
            self.index_trained = trained
            self._stale = int(sum(1 for i in ids.tolist() if i not in self.entries))
            self.stats['rebuilds'] += 1

    def _rebuild_in_background(self) -> None:
        if self._rebuild_thread is not None and self._rebuild_thread.is_alive():
            return
        self._rebuild_thread = threading.Thread(target=self.rebuild, daemon=True)
        self._rebuild_thread.start()

    def wait_for_rebuild(self) -> None:
        """Block until a background rebuild (if any) has swapped in its index."""
        if self._rebuild_thread is not None:
            self._rebuild_thread.join()

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
            metadata: Provenance: namespace, source (see provenance.TRUST), trust, timestamp, origin, ...

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
        if len(metadata_list) != n:
            raise ValueError(f"{n} embeddings but {len(metadata_list)} metadata records")
        if n > self.max_entries:
            raise ValueError(f"Batch of {n} entries exceeds store capacity {self.max_entries}")

        with self._lock:
            overflow = len(self.entries) + n - self.max_entries
            if overflow > 0:
                self._evict(max(overflow, int(0.1 * self.max_entries)))

            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n
            self.index.add_with_ids(vectors, ids)
            for i, entry_id in enumerate(ids.tolist()):
                metadata = {'namespace': 'default', 'source': 'model', **metadata_list[i]}
                metadata['trust'] = trust_level(metadata)
                self.entries[entry_id] = {'text': texts[i], 'vector': vectors[i:i + 1], 'metadata': metadata}
            self.stats['writes'] += n

            if self._needs_rebuild():
                self._rebuild_in_background()
            return ids.tolist()

    def _evict(self, n: int):
        """Tombstone the n oldest entries (FIFO)."""
        for _ in range(min(n, len(self.entries))):
            self.entries.popitem(last=False)
            self._stale += 1
            self.stats['evictions'] += 1

    def delete(self, ids: Iterable[int]) -> int:
        """Remove entries by id (tombstoned until the next rebuild). Returns the count removed."""
        removed = 0
        with self._lock:
            for entry_id in ids:
                if self.entries.pop(entry_id, None) is not None:
                    self._stale += 1
                    removed += 1
            self.stats['deletions'] += removed
            if removed and self._needs_rebuild():
                self._rebuild_in_background()
        return removed

    def delete_namespace(self, namespace: str) -> int:
        """Remove every entry of a namespace."""
        with self._lock:
            ids = [i for i, e in self.entries.items() if e['metadata'].get('namespace') == namespace]
        return self.delete(ids)

    def supersede(self, old_id: int, embedding, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Record an update: the new entry links to the old one, and the old
        entry stays as a source record but is no longer retrieved.
        """
        with self._lock:
            old = self.entries.get(old_id)
            if old is None:
                raise KeyError(f"No semantic entry {old_id}")
            new_id = self.add(embedding, text, {**(metadata or {}), 'supersedes': old_id})
            old['metadata']['superseded_by'] = new_id
            return new_id

    # ------------------------------------------------------------------- read

    def _search(self, query_embedding, top_k: int, accept: Callable[[int, Dict], bool],
                projected: bool = False) -> List[Tuple[int, Dict, float]]:
        """Nearest live entries satisfying `accept`, over-fetching in bounded steps."""
        vector = (np.ascontiguousarray(np.asarray(query_embedding, dtype='float32').reshape(1, -1))
                  if projected else self._vectors(query_embedding))
        with self._lock:
            self.stats['searches'] += 1
            if not self.entries or top_k <= 0:
                return []
            limit = min(self.index.ntotal, self.MAX_FETCH)
            k = min(limit, 4 * top_k + min(self._stale, 64))
            while True:
                distances, ids = self.index.search(vector, k)
                hits = []
                for entry_id, dist in zip(ids[0].tolist(), distances[0].tolist()):
                    entry = self.entries.get(entry_id)
                    if entry is not None and accept(entry_id, entry):
                        hits.append((entry_id, entry, dist))
                        if len(hits) == top_k:
                            return hits
                if k >= limit:
                    return hits
                k = min(limit, 2 * k)

    @staticmethod
    def _filter(namespaces: Optional[Iterable[str]], min_trust: Optional[int], exclude_ids: Iterable[int]) -> Callable:
        excluded = set(exclude_ids)
        allowed = set(namespaces) if namespaces is not None else None

        def accept(entry_id: int, entry: Dict) -> bool:
            meta = entry['metadata']
            return (entry_id not in excluded
                    and 'superseded_by' not in meta
                    and (allowed is None or meta.get('namespace') in allowed)
                    and (min_trust is None or meta.get('trust', 0) >= min_trust))
        return accept

    def retrieve(self, query_embedding, top_k: int = 5, return_distances: bool = False, **filters):
        """
        Retrieve most relevant fact texts (see retrieve_with_metadata for filters).

        Returns:
            List of fact texts, or (texts, distances)
        """
        hits = self.retrieve_with_metadata(query_embedding, top_k, **filters)
        texts = [h['text'] for h in hits]
        if return_distances:
            return texts, [h['distance'] for h in hits]
        return texts

    def retrieve_with_metadata(
        self,
        query_embedding,
        top_k: int = 5,
        namespaces: Optional[Iterable[str]] = None,
        min_trust: Optional[int] = None,
        exclude_ids: Iterable[int] = (),
    ) -> List[Dict]:
        """
        Args:
            query_embedding: (dimension,) base-model query embedding
            top_k: Number of results
            namespaces: Only entries of these namespaces (None = all)
            min_trust: Skip entries whose trust level is lower (None = all)
            exclude_ids: Entry ids to skip

        Returns:
            List of dicts with 'id', 'text', 'distance', 'metadata', 'vector'
        """
        return [
            {'id': entry_id, 'text': entry['text'], 'distance': dist, 'metadata': entry['metadata'],
             'vector': entry['vector'][0]}
            for entry_id, entry, dist in self._search(query_embedding, top_k, self._filter(namespaces, min_trust, exclude_ids))
        ]

    def size(self) -> int:
        """Return number of live entries."""
        with self._lock:
            return len(self.entries)

    def recall_at_k(self, query_embeddings, k: int = 10) -> float:
        """
        Fraction of exact top-k neighbours (brute force over live vectors)
        that the current index also returns. 1.0 for a flat index.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        queries = self._vectors(query_embeddings)
        with self._lock:
            if not self.entries:
                return 1.0
            ids = np.fromiter(self.entries.keys(), dtype=np.int64, count=len(self.entries))
            vectors = np.vstack([e['vector'] for e in self.entries.values()])
            k = min(k, len(ids))
            exact = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            exact.add_with_ids(vectors, ids)
            _, truth = exact.search(queries, k)
            found = 0
            for q in range(len(queries)):
                got = {i for i, _, _ in self._search(queries[q], k, lambda *_: True, projected=True)}
                found += len(got & set(truth[q].tolist()))
            return found / (k * len(queries))

    # ------------------------------------------------------------ persistence

    def save(self, path: str):
        """Save to `{path}.index` (FAISS) and `{path}.meta` (entries and counters)."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.wait_for_rebuild()
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
                    'stats': self.stats,
                    'projection': self.projection.cpu() if self.projection is not None else None,
                    'embedding_fingerprint': self.embedding_fingerprint,
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
            embedding_fingerprint=data.get('embedding_fingerprint'),
        )
        memory.entries = data['entries']
        memory._next_id = data['next_id']
        memory._stale = data['stale']
        memory.stats.update(data.get('stats', {}))
        memory.index_trained = data['index_trained']
        memory.index = memory._to_device(faiss.read_index(f"{path}.index"))
        print(f"Loaded semantic memory from {path} ({len(memory.entries)} entries)")
        return memory
