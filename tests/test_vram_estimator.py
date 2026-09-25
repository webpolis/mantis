from mantis.configs.model_config import get_tiny_config
from mantis.training.vram_estimator import estimate_training_vram


def test_optimizer_offload_frees_gpu_optimizer_state():
    cfg = get_tiny_config().base_moe
    on_gpu = estimate_training_vram(cfg, 256, 4, deepspeed_zero_stage=2, num_gpus=2)
    offloaded = estimate_training_vram(cfg, 256, 4, deepspeed_zero_stage=2, num_gpus=2, optimizer_offload=True)
    assert offloaded['optimizer_state'] == 0
    assert on_gpu['total'] - offloaded['total'] == on_gpu['optimizer_state']
