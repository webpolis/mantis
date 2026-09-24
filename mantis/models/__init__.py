"""
MANTIS Models Module

EpisodicMemorySSM and MambaBlock require mamba-ssm and load on first access.
"""

import importlib

from .base_moe import BaseMoEModel, MoELayer, Expert
from .meta_controller import MetaController, StateSummaryEncoder
from .critic import CriticModel, CriticValueNetwork

__all__ = [
    'BaseMoEModel',
    'MoELayer',
    'Expert',
    'MetaController',
    'StateSummaryEncoder',
    'CriticModel',
    'CriticValueNetwork',
    'EpisodicMemorySSM',
    'MambaBlock',
]


def __getattr__(name):
    if name in ('EpisodicMemorySSM', 'MambaBlock'):
        return getattr(importlib.import_module('mantis.models.ssm'), name)
    raise AttributeError(f"module 'mantis.models' has no attribute {name!r}")
