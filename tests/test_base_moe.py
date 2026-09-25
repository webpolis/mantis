import pytest
import torch

from tests.conftest import DEVICE, seed


@torch.no_grad()
def test_cached_decoding_matches_full_forward_with_gqa(base):
    assert base.layers[0].attn.n_kv_heads < base.layers[0].attn.n_heads
    seed()
    ids = torch.randint(4, 300, (1, 20), device=DEVICE)
    full = base(ids)['logits']
    past, steps = None, []
    for i in range(ids.size(1)):
        out = base(ids[:, i:i + 1], past_key_values=past, use_cache=True)
        past = out['past_key_values']
        steps.append(out['logits'])
    assert (full - torch.cat(steps, dim=1)).abs().max().item() < 1e-4
    key, value = past[0]
    assert key.shape[1] == base.layers[0].attn.n_kv_heads


@torch.no_grad()
def test_expert_weights_shape_is_checked(base):
    ids = torch.randint(4, 300, (1, 5), device=DEVICE)
    with pytest.raises(ValueError, match="expert_weights"):
        base(ids, expert_weights=torch.zeros(1, base.n_experts, device=DEVICE))
    zeros = torch.zeros(1, base.n_layers, base.n_experts, device=DEVICE)
    assert torch.allclose(base(ids)['logits'], base(ids, expert_weights=zeros)['logits'])


@torch.no_grad()
def test_expert_load_reports_topk_dispatch_per_layer(base):
    ids = torch.randint(4, 300, (2, 16), device=DEVICE)
    load = base(ids)['expert_load']
    assert load.shape == (base.n_layers, base.n_experts)
    assert torch.allclose(load.sum(dim=-1), torch.ones(base.n_layers, device=DEVICE))


@torch.no_grad()
def test_return_hidden_gives_only_final_hidden(base):
    out = base(torch.randint(4, 300, (1, 5), device=DEVICE), return_hidden=True)
    assert 'hidden_states' not in out
    assert out['last_hidden'].shape == (1, 5, base.d_model)
