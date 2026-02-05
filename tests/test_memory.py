"""
Unit tests for HMST memory systems.
"""

import torch
import pytest
import numpy as np
from hmst.models import EpisodicMemorySSM
from hmst.memory import EpisodicMemory, SemanticMemory


def test_episodic_memory():
    """Test EpisodicMemory storage and retrieval."""
    ssm = EpisodicMemorySSM(d_model=512, d_state=128, n_blocks=4)
    memory = EpisodicMemory(ssm, max_entries=10)

    # Add entries
    for i in range(5):
        tokens = torch.randint(0, 1000, (32,))
        embeddings = torch.randn(32, 512)
        memory.add(tokens, embeddings, metadata={'id': i})

    assert memory.size() == 5

    # Retrieve
    query_emb = torch.randn(512)
    results = memory.retrieve(query_emb, top_k=3)

    assert len(results) <= 3


def test_semantic_memory():
    """Test SemanticMemory storage and retrieval."""
    memory = SemanticMemory(
        dimension=128,
        max_entries=1000,
        index_type='Flat',
        use_gpu=False
    )

    # Add entries
    texts = [
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany"
    ]

    for text in texts:
        embedding = torch.randn(128)
        memory.add(embedding, text)

    assert memory.size() == 3

    # Retrieve
    query_emb = torch.randn(128)
    results = memory.retrieve(query_emb, top_k=2)

    assert len(results) <= 2
    assert all(isinstance(r, str) for r in results)


def test_semantic_memory_batch():
    """Test batch addition to semantic memory."""
    memory = SemanticMemory(
        dimension=128,
        max_entries=1000,
        index_type='Flat',
        use_gpu=False
    )

    # Batch add
    embeddings = torch.randn(10, 128)
    texts = [f"Fact {i}" for i in range(10)]

    memory.add_batch(embeddings, texts)

    assert memory.size() == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
