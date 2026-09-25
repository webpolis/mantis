"""
Shared fixtures: a seeded 2-layer tiny model (GQA on, 64-token window), the
tokenizer, and helpers to build engines with or without memory components.
"""

import dataclasses
import importlib.util

import pytest
import torch

from mantis.configs.model_config import get_tiny_config
from mantis.inference.engine import MANTISInferenceEngine, build_policy
from mantis.models.base_moe import BaseMoEModel
from mantis.models.critic import CriticModel
from mantis.tokenizer import MANTISTokenizer

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HAS_FAISS = importlib.util.find_spec('faiss') is not None
HAS_MAMBA = importlib.util.find_spec('mamba_ssm') is not None and torch.cuda.is_available()

needs_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
needs_mamba = pytest.mark.skipif(not HAS_MAMBA, reason="mamba_ssm needs CUDA")


def seed():
    torch.manual_seed(SEED)


@pytest.fixture(scope='session')
def config():
    cfg = get_tiny_config()
    cfg.base_moe.n_layers = 2
    cfg.base_moe.max_seq_len = 64
    cfg.critic.max_seq_len = 64
    cfg.meta_controller.n_layers = 1
    cfg.episodic_memory.n_blocks = 1
    cfg.episodic_memory.max_seq_len = 64
    cfg.inference.max_length = 6
    cfg.inference.temperature = 0.0
    cfg.validate()
    return cfg


@pytest.fixture(scope='session')
def tokenizer():
    return MANTISTokenizer()


@pytest.fixture(scope='session')
def base(config, tokenizer):
    config.base_moe.vocab_size = len(tokenizer)
    seed()
    return BaseMoEModel.from_config(config.base_moe).to(DEVICE).eval()


@pytest.fixture(scope='session')
def critic(config):
    seed()
    return CriticModel.from_config(config.critic, config.base_moe).to(DEVICE).eval()


@pytest.fixture(scope='session')
def ssm(config):
    if not HAS_MAMBA:
        pytest.skip("mamba_ssm needs CUDA")
    from mantis.models.ssm import EpisodicMemorySSM
    em = config.episodic_memory
    seed()
    return EpisodicMemorySSM(d_model=config.base_moe.d_model, d_state=em.d_state, n_blocks=em.n_blocks,
                             d_conv=em.d_conv, expand=em.expand, max_seq_len=em.max_seq_len).to(DEVICE).eval()


@pytest.fixture
def episodic(ssm, config):
    from mantis.memory.episodic import EpisodicMemory
    return EpisodicMemory(ssm, max_entries=4, context_window=config.base_moe.max_seq_len, device=DEVICE)


@pytest.fixture
def semantic(config):
    if not HAS_FAISS:
        pytest.skip("faiss not installed")
    from mantis.memory.semantic import SemanticMemory
    return SemanticMemory(dimension=config.base_moe.d_model, index_type='Flat', use_gpu=False)


@pytest.fixture
def make_engine(config, base, tokenizer):
    def build(episodic=None, semantic=None, critic=None, **overrides):
        inference = dataclasses.replace(config.inference, **overrides)
        seed()
        meta, encoder = build_policy(config)
        return MANTISInferenceEngine(base, meta, encoder, tokenizer, episodic, semantic, critic,
                                     config=inference, device=DEVICE)
    return build


@pytest.fixture
def random_hidden(config):
    def make(n_tokens: int):
        seed()
        return torch.randn(n_tokens, config.base_moe.d_model)
    return make
