"""
Semantic Memory System

Long-term knowledge storage using FAISS vector database.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss
import pickle
import os


class SemanticMemory:
    """
    Long-term knowledge storage using FAISS vector database.

    Stores facts with:
    - Dense vector embeddings
    - Metadata (source, confidence, timestamp)
    - Efficient approximate nearest neighbor search
    """

    def __init__(
        self,
        dimension: int = 1536,
        max_entries: int = 1_000_000,
        index_type: str = 'IVF',
        use_gpu: bool = True
    ):
        self.dimension = dimension
        self.max_entries = max_entries
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        # Initialize FAISS index
        self._init_index()

        # Metadata storage (parallel to FAISS index)
        # Each entry: {'text': str, 'embedding': np.ndarray, 'embedding_id': int, 'metadata': dict}
        self.metadata = []

        # Store embeddings for index training (IVF only)
        # Cleared after training to save memory, but embeddings are kept in self.metadata
        self.embeddings_cache = []

        # Temporary flat index for exact search before IVF is trained
        if self.index_type == 'IVF':
            self.pre_train_index = faiss.IndexFlatL2(self.dimension)
        else:
            self.pre_train_index = None

        # Sequential ID counter for IndexIDMap
        self._next_id = 0
        # Track stale (removed) entries for IVF deferred rebuild
        self._stale_count = 0

    def _init_index(self):
        """Initialize FAISS index based on type."""
        if self.index_type == 'Flat':
            # Exact search with ID map for efficient deletion
            base_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(base_index)

        elif self.index_type == 'IVF':
            # IVF with PQ compression (fast approximate search)
            n_clusters = min(16384, self.max_entries // 100)
            quantizer = faiss.IndexFlatL2(self.dimension)

            # PQ code size must divide dimension evenly
            code_size = min(128, self.dimension)
            while self.dimension % code_size != 0:
                code_size -= 1

            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                n_clusters,
                code_size,
                8     # bits per sub-quantizer
            )

            # Need training before use
            self.index_trained = False

        elif self.index_type == 'HNSW':
            # Hierarchical NSW graph with ID map for deletion support
            base_index = faiss.IndexHNSWFlat(self.dimension, 32)
            base_index.hnsw.efConstruction = 40
            base_index.hnsw.efSearch = 16
            self.index = faiss.IndexIDMap(base_index)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Failed to move to GPU: {e}. Using CPU.")
                self.use_gpu = False

    def add(
        self,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a new entry to semantic memory.

        Args:
            embedding: (dimension,) vector embedding
            text: Original text of the fact
            metadata: Optional dict with source, confidence, timestamp, etc.
        """
        # Handle capacity
        if len(self.metadata) >= self.max_entries:
            self._remove_oldest(int(0.1 * self.max_entries))

        # Ensure correct shape and type
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        embedding = embedding.astype('float32').reshape(1, -1)

        # Train index if needed (IVF requires training)
        if self.index_type == 'IVF' and not self.index_trained:
            if len(self.metadata) >= 10000:  # Minimum training size
                self._train_index()

        # Assign a unique ID to this entry
        entry_id = self._next_id
        self._next_id += 1

        # Add to index
        if self.index_type == 'IVF' and not self.index_trained:
            # IVF not trained yet: add to temporary flat index for exact search
            self.pre_train_index.add(embedding)
        elif self.index_type == 'IVF':
            self.index.add(embedding)
        else:
            # Flat/HNSW use IndexIDMap, require add_with_ids
            ids = np.array([entry_id], dtype=np.int64)
            self.index.add_with_ids(embedding, ids)

        # Store metadata with embedding for rebuilding
        entry = {
            'text': text,
            'embedding': embedding.copy(),
            'embedding_id': entry_id,
            'metadata': metadata or {}
        }
        self.metadata.append(entry)

        # Cache embedding for index training (IVF only, cleared after training)
        if self.index_type == 'IVF' and not self.index_trained:
            self.embeddings_cache.append(embedding.copy())

    def add_batch(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> None:
        """
        Add multiple entries at once (more efficient).

        Args:
            embeddings: (n, dimension) batch of embeddings
            texts: List of text strings
            metadata_list: Optional list of metadata dicts
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        embeddings = embeddings.astype('float32')
        n = embeddings.shape[0]

        if metadata_list is None:
            metadata_list = [{}] * n

        # Add to index
        if self.index_type == 'IVF' and not self.index_trained:
            # Collect training data
            if len(self.metadata) + n >= 10000:
                self._train_index()

        # Assign IDs for this batch
        batch_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n

        if self.index_type == 'IVF' and self.index_trained:
            self.index.add(embeddings)
        elif self.index_type == 'IVF' and not self.index_trained:
            if self.pre_train_index is not None:
                self.pre_train_index.add(embeddings)
        else:
            # Flat/HNSW use IndexIDMap
            self.index.add_with_ids(embeddings, batch_ids)

        # Store metadata with embeddings for rebuilding
        for i in range(n):
            entry = {
                'text': texts[i],
                'embedding': embeddings[i:i+1].copy(),
                'embedding_id': int(batch_ids[i]),
                'metadata': metadata_list[i]
            }
            self.metadata.append(entry)

        # Cache embeddings for index training (IVF only, cleared after training)
        if self.index_type == 'IVF' and not self.index_trained:
            for i in range(n):
                self.embeddings_cache.append(embeddings[i:i+1].copy())

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        return_distances: bool = False
    ) -> List[str]:
        """
        Retrieve most relevant facts.

        Args:
            query_embedding: (dimension,) query vector
            top_k: Number of results to return
            return_distances: Also return similarity scores

        Returns:
            List of text strings (facts), optionally with distances
        """
        if len(self.metadata) == 0:
            return [] if not return_distances else ([], [])

        # Prepare query
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Search (use pre-train index if IVF not yet trained)
        search_index = self.index
        if self.index_type == 'IVF' and not self.index_trained and self.pre_train_index is not None:
            search_index = self.pre_train_index

        try:
            distances, indices = search_index.search(query_embedding, min(top_k, len(self.metadata)))
        except Exception as e:
            print(f"Search failed: {e}")
            return [] if not return_distances else ([], [])

        # Retrieve corresponding text
        # For IndexIDMap (Flat/HNSW), search returns custom IDs, not positional indices
        # For IVF and pre_train_index, search returns positional indices
        uses_id_map = self.index_type in ('Flat', 'HNSW') and search_index is self.index

        results = []
        result_distances = []

        if uses_id_map:
            # Build ID -> metadata lookup
            id_to_entry = {entry['embedding_id']: entry for entry in self.metadata}
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx in id_to_entry:
                    results.append(id_to_entry[idx]['text'])
                    result_distances.append(float(dist))
        else:
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx]['text'])
                    result_distances.append(float(dist))

        if return_distances:
            return results, result_distances
        else:
            return results

    def retrieve_with_metadata(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve facts with full metadata using FAISS index positions directly.

        Returns:
            List of dicts with 'text', 'distance', 'metadata'
        """
        if len(self.metadata) == 0:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        search_index = self.index
        if self.index_type == 'IVF' and not self.index_trained and self.pre_train_index is not None:
            search_index = self.pre_train_index

        try:
            distances, indices = search_index.search(query_embedding, min(top_k, len(self.metadata)))
        except Exception as e:
            print(f"Search failed: {e}")
            return []

        uses_id_map = self.index_type in ('Flat', 'HNSW') and search_index is self.index

        results = []
        if uses_id_map:
            id_to_entry = {entry['embedding_id']: entry for entry in self.metadata}
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx in id_to_entry:
                    entry = id_to_entry[idx]
                    results.append({
                        'text': entry['text'],
                        'distance': float(dist),
                        'metadata': entry['metadata']
                    })
        else:
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and 0 <= idx < len(self.metadata):
                    entry = self.metadata[idx]
                    results.append({
                        'text': entry['text'],
                        'distance': float(dist),
                        'metadata': entry['metadata']
                    })

        return results

    def _train_index(self):
        """Train IVF index with current data."""
        if self.index_type != 'IVF' or self.index_trained:
            return

        print(f"Training FAISS index with {len(self.metadata)} samples...")

        # Use cached embeddings for training
        if len(self.embeddings_cache) > 0:
            training_data = np.vstack(self.embeddings_cache).astype('float32')
        else:
            print("Warning: No cached embeddings, cannot train index")
            return

        self.index.train(training_data)

        # Re-add all embeddings to the trained index
        self.index.add(training_data)

        self.index_trained = True

        # Clear training cache and pre-train index to save memory
        self.embeddings_cache = []
        self.pre_train_index = None

        print("Index training complete.")

    def _remove_oldest(self, n: int):
        """Remove n oldest entries (FIFO)."""
        removed_entries = self.metadata[:n]
        self.metadata = self.metadata[n:]

        if self.index_type == 'IVF':
            # IVF doesn't support efficient deletion via remove_ids
            # Track stale entries and only rebuild when >20% are stale
            self._stale_count += n
            total_indexed = self.index.ntotal if self.index_trained else 0
            if total_indexed > 0 and self._stale_count / total_indexed > 0.2:
                self._rebuild_index()
                self._stale_count = 0
        else:
            # Flat/HNSW use IndexIDMap which supports remove_ids
            ids_to_remove = np.array(
                [entry['embedding_id'] for entry in removed_entries],
                dtype=np.int64
            )
            self.index.remove_ids(ids_to_remove)

    def _rebuild_index(self):
        """Rebuild index from scratch (expensive operation)."""
        print(f"Rebuilding FAISS index with {len(self.metadata)} entries...")

        # Reinitialize index
        self._init_index()
        self.pre_train_index = faiss.IndexFlatL2(self.dimension) if self.index_type == 'IVF' else None
        self._stale_count = 0

        if len(self.metadata) == 0:
            print("Index rebuild complete (no entries to add).")
            return

        # Extract embeddings from metadata
        embeddings = np.vstack([entry['embedding'] for entry in self.metadata]).astype('float32')

        # Train index if needed (IVF)
        if self.index_type == 'IVF':
            if len(embeddings) >= 10000:
                print(f"  Training IVF index with {len(embeddings)} samples...")
                self.index.train(embeddings)
                self.index_trained = True
            else:
                print(f"  Warning: Only {len(embeddings)} samples, need >=10000 for training. Index remains untrained.")
                self.index_trained = False

        # Re-add all embeddings to the rebuilt index
        if self.index_type == 'IVF' and self.index_trained:
            self.index.add(embeddings)
            self.pre_train_index = None
            print(f"  Re-added {len(embeddings)} embeddings to index.")
        elif self.index_type == 'IVF':
            # Add to pre-train flat index for exact search until IVF is trained
            self.pre_train_index.add(embeddings)
            self.embeddings_cache = [embeddings[i:i+1].copy() for i in range(len(embeddings))]
            print(f"  Added {len(embeddings)} embeddings to pre-train index.")
        else:
            # Flat/HNSW use IndexIDMap with add_with_ids
            ids = np.array([entry['embedding_id'] for entry in self.metadata], dtype=np.int64)
            self.index.add_with_ids(embeddings, ids)
            print(f"  Re-added {len(embeddings)} embeddings to index.")

        print("Index rebuild complete.")

    def size(self) -> int:
        """Return number of entries."""
        return len(self.metadata)

    def save(self, path: str):
        """Save semantic memory to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save FAISS index
        faiss_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(faiss_index, f"{path}.index")

        # Save metadata
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'index_trained': getattr(self, 'index_trained', True)
            }, f)

        print(f"Saved semantic memory to {path}")

    def load(self, path: str):
        """Load semantic memory from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                self.use_gpu = False

        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.index_trained = data.get('index_trained', True)

        print(f"Loaded semantic memory from {path} ({len(self.metadata)} entries)")
