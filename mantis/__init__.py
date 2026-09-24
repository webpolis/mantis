"""
MANTIS: Metacognitive Adaptive Network with Tiered Inference Strategies

A novel LLM architecture for mitigating hallucination and extending long-context memory.

Exports load lazily so that importing the base model or tokenizer does not
require the optional memory dependencies (mamba-ssm, faiss).
"""

import importlib

__version__ = '1.0.0'

_EXPORTS = {
    'BaseMoEModel': 'mantis.models.base_moe',
    'MetaController': 'mantis.models.meta_controller',
    'CriticModel': 'mantis.models.critic',
    'EpisodicMemorySSM': 'mantis.models.ssm',
    'EpisodicMemory': 'mantis.memory.episodic',
    'SemanticMemory': 'mantis.memory.semantic',
    'MemoryConsolidator': 'mantis.memory.consolidation',
    'MANTISInferenceEngine': 'mantis.inference.engine',
    'MANTISConfig': 'mantis.configs.model_config',
    'get_micro_config': 'mantis.configs.model_config',
    'get_tiny_config': 'mantis.configs.model_config',
    'get_small_config': 'mantis.configs.model_config',
    'get_base_config': 'mantis.configs.model_config',
    'get_large_config': 'mantis.configs.model_config',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'mantis' has no attribute {name!r}")
