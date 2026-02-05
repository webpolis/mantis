"""
HMST Model Configuration

Centralized configuration for all model components.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseMoEConfig:
    """Configuration for Base MoE Model."""
    vocab_size: int = 128000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    d_ff: int = 8192
    n_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 8192
    dropout: float = 0.1
    load_balance_weight: float = 0.01


@dataclass
class MetaControllerConfig:
    """Configuration for Meta-Controller."""
    d_model: int = 1024
    n_layers: int = 6
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.1
    n_experts: int = 8
    state_dim: int = 128


@dataclass
class CriticConfig:
    """Configuration for Critic Model."""
    vocab_size: int = 128000
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.1


@dataclass
class EpisodicMemoryConfig:
    """Configuration for Episodic Memory (SSM)."""
    d_model: int = 2048
    d_state: int = 256
    n_blocks: int = 8
    d_conv: int = 4
    expand: int = 2
    max_seq_len: int = 8192
    dropout: float = 0.1
    max_entries: int = 100


@dataclass
class SemanticMemoryConfig:
    """Configuration for Semantic Memory (FAISS)."""
    dimension: int = 1536
    max_entries: int = 1_000_000
    index_type: str = 'IVF'  # 'Flat', 'IVF', 'HNSW'
    use_gpu: bool = True


@dataclass
class TrainingConfig:
    """Configuration for Training."""

    # Stage 1: Pre-training
    pretrain_lr: float = 3e-4
    pretrain_batch_size: int = 2048
    pretrain_steps: int = 100000
    pretrain_warmup: int = 2000
    pretrain_weight_decay: float = 0.01
    pretrain_gradient_clip: float = 1.0

    # Stage 2: Memory fine-tuning
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 512
    finetune_steps: int = 20000
    finetune_warmup: int = 1000

    # Stage 3: RL training
    rl_lr: float = 1e-5
    rl_batch_size: int = 256
    rl_episodes: int = 50000
    rl_ppo_epsilon: float = 0.2
    rl_discount: float = 0.99
    rl_gae_lambda: float = 0.95

    # Reward weights
    alpha_accuracy: float = 1.0
    beta_latency: float = 0.3
    gamma_compute: float = 0.2
    delta_calibration: float = 0.5

    # General
    seed: int = 42
    use_mixed_precision: bool = True
    use_wandb: bool = False
    checkpoint_dir: str = './checkpoints'
    log_interval: int = 100
    save_interval: int = 10000


@dataclass
class InferenceConfig:
    """Configuration for Inference."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    routing_threshold: float = 0.5
    early_exit_uncertainty_threshold: float = 0.2
    verification_confidence_threshold: float = 0.6
    device: str = 'cuda'


@dataclass
class HMSTConfig:
    """Complete HMST Configuration."""
    base_moe: BaseMoEConfig = field(default_factory=BaseMoEConfig)
    meta_controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    episodic_memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    semantic_memory: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def save(self, path: str):
        """Save configuration to file."""
        import json
        from dataclasses import asdict

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            base_moe=BaseMoEConfig(**data['base_moe']),
            meta_controller=MetaControllerConfig(**data['meta_controller']),
            critic=CriticConfig(**data['critic']),
            episodic_memory=EpisodicMemoryConfig(**data['episodic_memory']),
            semantic_memory=SemanticMemoryConfig(**data['semantic_memory']),
            training=TrainingConfig(**data['training']),
            inference=InferenceConfig(**data['inference'])
        )


# Default configurations for different scales
def get_tiny_config() -> HMSTConfig:
    """Tiny model for rapid testing (~100M parameters)."""
    config = HMSTConfig()
    config.base_moe.d_model = 512
    config.base_moe.n_layers = 6
    config.base_moe.n_heads = 8
    config.base_moe.d_ff = 2048
    config.base_moe.n_experts = 4
    config.base_moe.top_k = 2
    config.meta_controller.n_experts = 4  # Must match base_moe.n_experts
    return config


def get_small_config() -> HMSTConfig:
    """Small model for testing (~1B parameters)."""
    config = HMSTConfig()
    config.base_moe.d_model = 1024
    config.base_moe.n_layers = 12
    config.base_moe.d_ff = 4096
    config.base_moe.n_experts = 4
    config.meta_controller.n_experts = 4  # Must match base_moe.n_experts
    return config


def get_base_config() -> HMSTConfig:
    """Base model (~12B parameters)."""
    return HMSTConfig()


def get_large_config() -> HMSTConfig:
    """Large model (~30B parameters)."""
    config = HMSTConfig()
    config.base_moe.d_model = 4096
    config.base_moe.n_layers = 32
    config.base_moe.d_ff = 16384
    config.base_moe.n_experts = 16
    return config
