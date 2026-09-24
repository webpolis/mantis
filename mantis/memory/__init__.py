"""
MANTIS Memory Systems Module

Episodic memory requires mamba-ssm and semantic memory requires faiss; each
loads on first access.
"""

import importlib

_EXPORTS = {
    'EpisodicMemory': 'mantis.memory.episodic',
    'group_similar': 'mantis.memory.episodic',
    'SemanticMemory': 'mantis.memory.semantic',
    'MemoryConsolidator': 'mantis.memory.consolidation',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'mantis.memory' has no attribute {name!r}")
