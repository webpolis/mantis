"""
MANTIS: Metacognitive Adaptive Network with Tiered Inference Strategies

A novel LLM architecture for mitigating hallucination and extending long-context memory.
"""

__version__ = '1.0.0'

from .models import (
    BaseMoEModel,
    MetaController,
    CriticModel,
    EpisodicMemorySSM
)

from .memory import (
    EpisodicMemory,
    SemanticMemory,
    MemoryConsolidator
)

from .inference import MANTISInferenceEngine

from .configs.model_config import (
    MANTISConfig,
    get_micro_config,
    get_tiny_config,
    get_small_config,
    get_base_config,
    get_large_config
)

__all__ = [
    'BaseMoEModel',
    'MetaController',
    'CriticModel',
    'EpisodicMemorySSM',
    'EpisodicMemory',
    'SemanticMemory',
    'MemoryConsolidator',
    'MANTISInferenceEngine',
    'MANTISConfig',
    'get_micro_config',
    'get_tiny_config',
    'get_small_config',
    'get_base_config',
    'get_large_config',
]
