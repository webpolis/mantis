"""
MANTIS Model Configuration

Centralized configuration for all model components.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseMoEConfig:
    """Configuration for Base MoE Model."""
    vocab_size: int = 512
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
    """Configuration for Meta-Controller (residual MLP policy)."""
    d_model: int = 2048  # Must match BaseMoEConfig.d_model for input alignment
    n_layers: int = 6
    d_ff: int = 4096
    dropout: float = 0.1
    n_experts: int = 8
    state_dim: int = 128


@dataclass
class CriticConfig:
    """Configuration for Critic Model."""
    vocab_size: int = 512
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
    dimension: int = 2048  # Must match BaseMoEConfig.d_model
    max_entries: int = 1_000_000
    index_type: str = 'IVF'  # 'Flat', 'IVF', 'HNSW'
    use_gpu: bool = True


@dataclass
class TrainingConfig:
    """Stage 2/3 defaults (Stage 1 is configured from the train.py CLI)."""

    # Stage 2: Memory fine-tuning
    finetune_lr: float = 1e-5
    episodic_loss_weight: float = 1.0
    semantic_loss_weight: float = 0.5

    # Stage 3: RL training
    rl_lr: float = 1e-5
    rl_ppo_epsilon: float = 0.2

    # Reward weights
    alpha_accuracy: float = 1.0
    beta_latency: float = 0.3
    gamma_compute: float = 0.2
    delta_calibration: float = 0.5

    # Critic training
    critic_lr: float = 1e-4


@dataclass
class InferenceConfig:
    """Configuration for the full MANTIS inference engine."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    routing_threshold: float = 0.5
    early_exit_uncertainty_threshold: float = 0.2
    verification_confidence_threshold: float = 0.6


@dataclass
class MANTISConfig:
    """Complete MANTIS Configuration."""
    base_moe: BaseMoEConfig = field(default_factory=BaseMoEConfig)
    meta_controller: MetaControllerConfig = field(
        default_factory=MetaControllerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    episodic_memory: EpisodicMemoryConfig = field(
        default_factory=EpisodicMemoryConfig)
    semantic_memory: SemanticMemoryConfig = field(
        default_factory=SemanticMemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check cross-component alignment. Call again after mutating a config."""
        d = self.base_moe.d_model
        checks = [
            (self.meta_controller.d_model == d, f"MetaController d_model ({self.meta_controller.d_model})"),
            (self.semantic_memory.dimension == d, f"SemanticMemory dimension ({self.semantic_memory.dimension})"),
            (self.episodic_memory.d_model == d, f"EpisodicMemory d_model ({self.episodic_memory.d_model})"),
        ]
        for ok, what in checks:
            if not ok:
                raise ValueError(f"{what} must match BaseMoE d_model ({d})")
        if self.meta_controller.n_experts != self.base_moe.n_experts:
            raise ValueError(
                f"MetaController n_experts ({self.meta_controller.n_experts}) must match "
                f"BaseMoE n_experts ({self.base_moe.n_experts})"
            )
        if d % self.base_moe.n_heads != 0 or (d // self.base_moe.n_heads) % 2 != 0:
            raise ValueError(f"d_model ({d}) / n_heads ({self.base_moe.n_heads}) must be an even integer for RoPE")
        if self.base_moe.top_k > self.base_moe.n_experts:
            raise ValueError(f"top_k ({self.base_moe.top_k}) cannot exceed n_experts ({self.base_moe.n_experts})")

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
def _sized_config(d_model: int, n_layers: int, n_heads: int, d_ff: int,
                  n_experts: int, top_k: int) -> MANTISConfig:
    """Build a preset with every d_model / n_experts-dependent field aligned."""
    config = MANTISConfig()
    config.base_moe.d_model = d_model
    config.base_moe.n_layers = n_layers
    config.base_moe.n_heads = n_heads
    config.base_moe.d_ff = d_ff
    config.base_moe.n_experts = n_experts
    config.base_moe.top_k = top_k
    config.meta_controller.d_model = d_model
    config.meta_controller.n_experts = n_experts
    config.semantic_memory.dimension = d_model
    config.episodic_memory.d_model = d_model
    config.validate()
    return config


def get_micro_config() -> MANTISConfig:
    """Micro model for TinyStories dataset (10-15M parameters, dense)."""
    return _sized_config(d_model=256, n_layers=4, n_heads=4, d_ff=1024, n_experts=1, top_k=1)


def get_tiny_config() -> MANTISConfig:
    """Tiny model for rapid testing (~100M parameters)."""
    return _sized_config(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_experts=4, top_k=2)


def get_small_config() -> MANTISConfig:
    """Small model for testing (~1B parameters)."""
    return _sized_config(d_model=1024, n_layers=12, n_heads=32, d_ff=4096, n_experts=4, top_k=2)


def get_base_config() -> MANTISConfig:
    """Base model (~12B parameters)."""
    return MANTISConfig()


def get_large_config() -> MANTISConfig:
    """Large model (~30B parameters)."""
    return _sized_config(d_model=4096, n_layers=32, n_heads=32, d_ff=16384, n_experts=16, top_k=2)


def get_extmem_config() -> MANTISConfig:
    """Extended episodic memory"""
    config = MANTISConfig()
    config.episodic_memory.max_seq_len = 32768  # 32K tokens
    config.base_moe.max_seq_len = 32768  # Match the base model too
    config.validate()
    return config
