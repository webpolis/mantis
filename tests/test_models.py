"""
Unit tests for HMST models.
"""

import torch
import pytest
from hmst.models import BaseMoEModel, MetaController, CriticModel, EpisodicMemorySSM


def test_base_moe_model():
    """Test BaseMoEModel forward pass."""
    model = BaseMoEModel(
        vocab_size=1000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        d_ff=2048,
        n_experts=4,
        max_seq_len=128
    )

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    output = model(input_ids)

    assert 'logits' in output
    assert output['logits'].shape == (batch_size, seq_len, 1000)
    assert 'load_balance_loss' in output


def test_meta_controller():
    """Test MetaController routing decisions."""
    controller = MetaController(
        d_model=512,
        n_layers=3,
        n_heads=8,
        d_ff=1024,
        n_experts=4,
        state_dim=64
    )

    batch_size = 2
    query_emb = torch.randn(batch_size, 512)
    state_summary = torch.randn(batch_size, 64)

    decisions = controller(query_emb, state_summary)

    assert 'early_exit' in decisions
    assert 'episodic' in decisions
    assert 'semantic' in decisions
    assert 'verification' in decisions
    assert 'expert_weights' in decisions
    assert 'uncertainty' in decisions

    # Check shapes
    assert decisions['early_exit'].shape == (batch_size, 1)
    assert decisions['expert_weights'].shape == (batch_size, 4)


def test_critic_model():
    """Test CriticModel verification."""
    critic = CriticModel(
        vocab_size=1000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=128
    )

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    correctness, confidence = critic(input_ids, segment_ids)

    assert correctness.shape == (batch_size, 1)
    assert confidence.shape == (batch_size, 1)
    assert (correctness >= 0).all() and (correctness <= 1).all()


def test_episodic_memory_ssm():
    """Test EpisodicMemorySSM encoding."""
    ssm = EpisodicMemorySSM(
        d_model=512,
        d_state=128,
        n_blocks=4,
        max_seq_len=256
    )

    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, 512)

    output, state = ssm(x, return_state=True)

    assert output.shape == (batch_size, seq_len, 512)
    assert state.shape == (batch_size, 128)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
