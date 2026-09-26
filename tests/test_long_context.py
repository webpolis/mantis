import dataclasses
import math

import pytest
import torch

from mantis.configs.model_config import MANTISConfig, get_tiny_config
from mantis.models.base_moe import BaseMoEModel, CausalSelfAttention, RotaryEmbedding, apply_rotary
from mantis.utils.checkpoints import migrate_config
from tests.conftest import DEVICE, seed


def test_yarn_ramp_and_temperature():
    plain = RotaryEmbedding(64)
    yarn = RotaryEmbedding(64, factor=512, original_context=2048)
    assert plain.attention_scale is None
    assert yarn.inv_freq[:9] == plain.inv_freq[:9]                      # pairs 0-8 keep the fast frequencies
    for i in range(21, 32):                                              # pairs 21-31 are fully interpolated
        assert math.isclose(yarn.inv_freq[i], plain.inv_freq[i] / 512, rel_tol=1e-12)
    assert plain.inv_freq[12] / 512 < yarn.inv_freq[12] < plain.inv_freq[12]
    assert math.isclose(yarn.attention_scale, (1 + 0.1 * math.log(512)) ** 2 / 8, rel_tol=1e-9)
    assert math.isclose(RotaryEmbedding(64, factor=128).attention_scale, 0.275729, rel_tol=1e-5)
    assert RotaryEmbedding(64, factor=1.0).inv_freq == plain.inv_freq


def test_rope_phases_stay_exact_at_a_million_positions():
    rope = RotaryEmbedding(64)
    positions = torch.tensor([0, 2048, 1_048_576])
    cos, sin = rope(positions)
    inv = torch.tensor(rope.inv_freq, dtype=torch.float64)
    ref = (positions.double()[:, None] * inv[None, :]).cos()
    assert (cos.double() - ref).abs().max() < 1e-6
    assert cos.dtype == torch.float32 and sin.shape == (3, 32)


def _dense_window_reference(attn, x, rope, window, past_len=0):
    """Attention over `x` with an explicit boolean sliding-window mask."""
    batch, seq_len, _ = x.shape
    q = attn.q_proj(x).view(batch, seq_len, attn.n_heads, attn.d_head).transpose(1, 2)
    k, v = attn.kv_proj(x).view(batch, seq_len, 2, attn.n_kv_heads, attn.d_head).permute(2, 0, 3, 1, 4)
    q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
    pos = torch.arange(seq_len, device=x.device)
    distance = pos[:, None] - pos[None, :]
    mask = (distance >= 0) & (distance < window)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scale, enable_gqa=True)
    return attn.out(out.transpose(1, 2).reshape(batch, seq_len, -1))


def test_local_attention_matches_masked_dense_with_gradients():
    seed()
    attn = CausalSelfAttention(256, n_heads=8, n_kv_heads=4, dropout=0.0, window=16, scale=0.2).to(DEVICE)
    rope = RotaryEmbedding(32)(torch.arange(48, device=DEVICE))
    x = torch.randn(2, 48, 256, device=DEVICE, requires_grad=True)
    out, _ = attn(x, rope, None, None, False)
    ref = _dense_window_reference(attn, x, rope, 16)
    assert torch.allclose(out, ref, atol=1e-4)
    grad_flex, = torch.autograd.grad(out.sum(), x, retain_graph=True)
    grad_ref, = torch.autograd.grad(ref.sum(), x)
    assert torch.allclose(grad_flex, grad_ref, atol=1e-4)


def test_local_layer_cache_keeps_a_window_and_matches_full_prefill():
    seed()
    attn = CausalSelfAttention(256, n_heads=8, n_kv_heads=4, dropout=0.0, window=8).to(DEVICE).eval()
    positions = torch.arange(40, device=DEVICE)
    rope = RotaryEmbedding(32)
    x = torch.randn(1, 40, 256, device=DEVICE)
    full, _ = attn(x, rope(positions), None, None, False)
    # prefill in two chunks, then one token at a time
    out1, past = attn(x[:, :13], rope(positions[:13]), None, None, True)
    out2, past = attn(x[:, 13:30], rope(positions[13:30]), None, past, True, past_len=13)
    assert past[0].size(2) == 7
    steps = []
    for t in range(30, 40):
        o, past = attn(x[:, t:t + 1], rope(positions[t:t + 1]), None, past, True, past_len=t)
        steps.append(o)
    chunked = torch.cat([out1, out2] + steps, dim=1)
    assert torch.allclose(full, chunked, atol=1e-4)


def _model(config, **overrides):
    cfg = dataclasses.replace(config.base_moe, dropout=0.0, **overrides)
    seed()
    return BaseMoEModel.from_config(cfg).to(DEVICE).eval()


def test_layout_is_the_legacy_model_up_to_the_window(config):
    legacy = _model(config)
    local = _model(config, local_window=64, global_layers=(1,), rope_factor=1.0)
    local.load_state_dict(legacy.state_dict())
    assert local.attention_layout() == (64, (1,), 1.0, 2048) and legacy.attention_layout() == (0, (), 1.0, 2048)
    ids = torch.randint(4, 300, (2, 48), device=DEVICE)
    assert torch.allclose(legacy(ids)['logits'], local(ids)['logits'], atol=1e-4)


def test_cached_decoding_with_local_and_global_layers_matches_full_forward(config):
    model = _model(config, local_window=8, global_layers=(1,), rope_factor=16.0)
    ids = torch.randint(4, 300, (1, 40), device=DEVICE)
    full = model(ids)['logits']
    out = model(ids[:, :20], use_cache=True)
    past, steps = out['past_key_values'], [out['logits']]
    assert past.length == 20 and past[0][0].size(2) == 7 and past[1][0].size(2) == 20
    for t in range(20, 40):
        out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = out['past_key_values']
        steps.append(out['logits'])
    assert past.length == 40 and model.cache_len(past) == 40
    assert (full - torch.cat(steps, dim=1)).abs().max().item() < 1e-3


def test_config_validation_and_migration():
    cfg = get_tiny_config()
    cfg.base_moe.local_window = 1024
    cfg.base_moe.global_layers = (2, 5)
    cfg.base_moe.rope_factor = 64.0
    cfg.validate()
    for bad in ({'global_layers': (6,)}, {'rope_factor': 0.5}, {'local_window': 0}):
        broken = get_tiny_config()
        broken.base_moe.local_window = 1024
        broken.base_moe.global_layers = (2,)
        for k, v in bad.items():
            setattr(broken.base_moe, k, v)
        with pytest.raises(ValueError):
            broken.validate()
    old = get_tiny_config()
    for name in ('local_window', 'global_layers', 'rope_factor', 'rope_original_context'):
        delattr(old.base_moe, name)
    migrate_config(old)
    assert (old.base_moe.local_window, old.base_moe.global_layers, old.base_moe.rope_factor) == (0, (), 1.0)
    old.validate()


def test_window_block_mask_matches_dense_definition():
    from torch.nn.attention.flex_attention import flex_attention
    from mantis.models.base_moe import _window_mask
    seed()
    for seq_len, total, past_len, window in ((300, 300, 0, 100), (17, 24, 13, 8), (1000, 1200, 700, 256)):
        q = torch.randn(1, 4, seq_len, 32, device=DEVICE)
        k = torch.randn(1, 4, total, 32, device=DEVICE)
        v = torch.randn_like(k)
        qi = past_len + torch.arange(seq_len, device=DEVICE)[:, None]
        ki = torch.arange(total, device=DEVICE)[None, :] + (past_len - (total - seq_len))
        dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=((qi - ki) >= 0) & ((qi - ki) < window))
        mask = _window_mask(seq_len, total, past_len, window, torch.device(DEVICE))
        assert torch.allclose(flex_attention(q, k, v, block_mask=mask), dense, atol=1e-5)
