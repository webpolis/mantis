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
        self.metadata = []

        # Store embeddings for index training/rebuilding
        self.embeddings_cache = []

    def _init_index(self):
        """Initialize FAISS index based on type."""
        if self.index_type == 'Flat':
            # Exact search (slow but accurate)
            self.index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == 'IVF':
            # IVF with PQ compression (fast approximate search)
            n_clusters = min(16384, self.max_entries // 100)
            quantizer = faiss.IndexFlatL2(self.dimension)

            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                n_clusters,
                128,  # code size
                8     # bits per sub-quantizer
            )

            # Need training before use
            self.index_trained = False

        elif self.index_type == 'HNSW':
            # Hierarchical NSW graph (very fast, moderate memory)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16

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

        # Add to index
        if self.index_type == 'IVF' and not self.index_trained:
            # Store for later training
            pass  # Will train when enough samples
        else:
            self.index.add(embedding)

        # Store metadata
        entry = {
            'text': text,
            'embedding_id': len(self.metadata),
            'metadata': metadata or {}
        }
        self.metadata.append(entry)

        # Cache embedding for index training
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

        if self.index_type != 'IVF' or self.index_trained:
            self.index.add(embeddings)

        # Store metadata
        for i in range(n):
            entry = {
                'text': texts[i],
                'embedding_id': len(self.metadata),
                'metadata': metadata_list[i]
            }
            self.metadata.append(entry)

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

        # Search
        try:
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        except Exception as e:
            print(f"Search failed: {e}")
            return [] if not return_distances else ([], [])

        # Retrieve corresponding text
        results = []
        result_distances = []

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
        Retrieve facts with full metadata.

        Returns:
            List of dicts with 'text', 'distance', 'metadata'
        """
        texts, distances = self.retrieve(
            query_embedding,
            top_k,
            return_distances=True
        )

        results = []
        for text, dist in zip(texts, distances):
            # Find metadata
            for entry in self.metadata:
                if entry['text'] == text:
                    results.append({
                        'text': text,
                        'distance': dist,
                        'metadata': entry['metadata']
                    })
                    break

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

        # Clear cache to prevent memory leak (embeddings are now in FAISS index)
        self.embeddings_cache = []

        print("Index training complete.")

    def _remove_oldest(self, n: int):
        """Remove n oldest entries (FIFO)."""
        # Remove from metadata
        self.metadata = self.metadata[n:]

        # Rebuild index (FAISS doesn't support efficient deletion)
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild index from scratch (expensive operation)."""
        print("Rebuilding FAISS index...")

        old_index_type = self.index_type
        self._init_index()

        # Re-add all entries (would need cached embeddings in production)
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
