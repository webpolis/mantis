
import pytest
import torch
import torch.nn.functional as F

from mantis.configs.model_config import get_base_config, get_micro_config, get_tiny_config
from mantis.inference.engine import build_policy
from mantis.models.critic import CriticModel
from mantis.models.meta_controller import EXPERT, GATES
from tests.conftest import DEVICE, seed


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def test_controller_expert_bias_is_zero_at_init_and_bounded(config):
    seed()
    meta, encoder = build_policy(config)
    meta.eval()
    query = torch.randn(2, config.base_moe.d_model)
    state = torch.randn(2, config.meta_controller.state_dim)
    decisions = meta(query, state)
    assert torch.all(meta.expert_bias(decisions['expert_raw']) == 0)
    bias = meta.expert_bias(torch.randn(2, config.base_moe.n_layers * config.base_moe.n_experts) * 100)
    assert bias.shape == (2, config.base_moe.n_layers, config.base_moe.n_experts)
    assert bias.abs().max() <= config.meta_controller.expert_bias_scale


def test_controller_action_mask_excludes_unused_actions(config):
    seed()
    meta, _ = build_policy(config)
    meta.eval()
    decisions = meta(torch.randn(1, config.base_moe.d_model), torch.randn(1, config.meta_controller.state_dim))
    gate_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    actions, raw = meta.act(decisions, gate_mask, sample=False)
    assert actions.shape == (1, len(GATES)) and set(actions.unique().tolist()) <= {0.0, 1.0}

    bypass_only = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    gate0 = torch.distributions.Bernoulli(logits=decisions['gate_logits'][:, 0]).log_prob(actions[:, 0])
    assert torch.allclose(meta.log_prob(decisions, actions, raw, bypass_only), gate0)

    with_expert = bypass_only.clone()
    with_expert[:, EXPERT] = 1.0
    assert meta.log_prob(decisions, actions, raw, with_expert).item() != gate0.item()

    masked_gate = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])
    masked_actions, _ = meta.act(decisions, masked_gate[:, :EXPERT], sample=False)
    assert masked_actions[0, 1] == 0


def test_critic_build_input_order_and_budget(critic):
    ids, segs = critic.build_input(query_ids=[1] * 5, response_ids=[2] * 100, evidence_ids=[3] * 10)
    assert len(ids) == critic.max_seq_len == 64
    assert ids[:10] == [3] * 10 and ids[10:15] == [1] * 5 and ids[15:] == [2] * 49
    assert segs == [0] * 10 + [1] * 5 + [2] * 49

    ids, segs = critic.build_input([1] * 30, [2] * 10, [3] * 100)
    counts = {tok: ids.count(tok) for tok in (3, 1, 2)}
    assert counts == {3: 45, 1: 9, 2: 10}  # shares 22/9/10, then leftover 23 extends the evidence
    assert len(ids) == 64 and segs[-10:] == [2] * 10


def test_critic_temperature_scaling_lowers_nll(critic):
    seed()
    z = torch.randn(2000)
    targets = torch.bernoulli(torch.sigmoid(z))
    logits = (5.0 * z).unsqueeze(1)
    before = F.binary_cross_entropy_with_logits(logits, targets.unsqueeze(1))
    t = critic.fit_temperature(logits.to(DEVICE), targets.unsqueeze(1).to(DEVICE))
    after = F.binary_cross_entropy_with_logits(logits / t, targets.unsqueeze(1))
    assert 3.0 < t < 8.0 and after < before
    critic.temperature.fill_(1.0)


@torch.no_grad()
def test_critic_verify_returns_probability(critic, base):
    score = critic.verify(base, [5, 6, 7], [8, 9, 10], [11, 12])
    assert 0.0 <= score <= 1.0


def test_config_validation_rejects_misalignment():
    cfg = get_tiny_config()
    cfg.base_moe.n_kv_heads = 3
    with pytest.raises(ValueError, match="n_kv_heads"):
        cfg.validate()
    cfg = get_tiny_config()
    cfg.critic.max_seq_len = cfg.base_moe.max_seq_len + 1
    with pytest.raises(ValueError, match="Critic max_seq_len"):
        cfg.validate()
    cfg = get_tiny_config()
    cfg.inference.route_policy = 'random'
    with pytest.raises(ValueError, match="route_policy"):
        cfg.validate()
    cfg = get_tiny_config()
    cfg.inference.prompt_format = 'xml'
    with pytest.raises(ValueError, match="prompt_format"):
        cfg.validate()


def test_presets_scale_auxiliary_models():
    sizes = {}
    for name, get in (('micro', get_micro_config), ('tiny', get_tiny_config), ('base', get_base_config)):
        cfg = get()
        meta, _ = build_policy(cfg)
        sizes[name] = (n_params(meta), n_params(CriticModel.from_config(cfg.critic, cfg.base_moe)))
    assert sizes['micro'][0] < sizes['tiny'][0] < sizes['base'][0]
    assert sizes['micro'][1] < sizes['tiny'][1] < sizes['base'][1]
    assert sizes['micro'][1] < 3_000_000 and sizes['micro'][0] < 1_000_000
    micro = get_micro_config()
    assert micro.base_moe.n_kv_heads == micro.base_moe.n_heads
    assert get_base_config().base_moe.n_kv_heads == 8
