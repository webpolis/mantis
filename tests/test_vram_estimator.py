from mantis.configs.model_config import get_tiny_config
from mantis.training.vram_estimator import estimate_training_vram, plan_layer_placement


def test_optimizer_offload_frees_gpu_optimizer_state():
    cfg = get_tiny_config().base_moe
    on_gpu = estimate_training_vram(cfg, 256, 4, deepspeed_zero_stage=2, num_gpus=2)
    offloaded = estimate_training_vram(cfg, 256, 4, deepspeed_zero_stage=2, num_gpus=2, optimizer_offload=True)
    assert offloaded['optimizer_state'] == 0
    assert offloaded['optimizer_step'] == 0
    assert on_gpu['total'] - offloaded['total'] == on_gpu['optimizer_state'] + on_gpu['optimizer_step']


def test_per_device_split_adds_up_to_the_total():
    cfg = get_tiny_config().base_moe
    for ckpt in (False, True):
        est = estimate_training_vram(cfg, 256, 3, gradient_checkpointing=ckpt)
        layers = cfg.n_layers * (est['layer_fixed'] + est['layer_per_sample'] * 3)
        head = est['head_fixed'] + est['head_per_sample'] * 3
        assert layers + head + est['cuda_overhead'] + est['recompute_per_sample'] * 3 == est['total']


def test_layer_placement_fills_devices_in_order():
    cfg = get_tiny_config().base_moe
    est = estimate_training_vram(cfg, 256, 2)
    layer = est['layer_fixed'] + est['layer_per_sample'] * 2
    head = est['cuda_overhead'] + est['head_fixed'] + est['head_per_sample'] * 2
    # Device 0 holds the head plus three layers, device 1 the rest
    free = [(head + 3.5 * layer) / 0.85, (est['cuda_overhead'] + cfg.n_layers * layer) / 0.85]
    placement = plan_layer_placement(cfg, 256, 2, free)
    assert placement == [0, 0, 0] + [1] * (cfg.n_layers - 3)
    assert plan_layer_placement(cfg, 256, 2, [free[0]]) is None
    assert plan_layer_placement(cfg, 256, 2, [layer / 2] * 100) is None
