"""
Shared setup for the Stage 1 trainers (train.py and train_evo.py).
"""

import importlib.util
import math
import sys

import torch


def build_accelerator(args):
    """
    Create the Accelerator.

    - DeepSpeed ZeRO-2 when --deepspeed is set and at least two GPUs are visible.
    - DDP runs with find_unused_parameters=True: an expert that receives no
      tokens in a micro-batch gets no gradient, which plain DDP rejects.
    - Seedable samplers make each epoch's shuffle order reproducible, so a
      mid-epoch checkpoint can resume at the exact batch.

    Returns:
        (accelerator, use_deepspeed)
    """
    try:
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
    except ModuleNotFoundError as exc:
        if exc.name != 'accelerate':
            raise
        raise RuntimeError(
            f"accelerate is missing from {sys.executable}. Install it in the Python environment "
            "used to launch torchrun (python -m pip install accelerate)."
        ) from exc

    deepspeed_plugin = None
    if args.deepspeed:
        if torch.cuda.device_count() < 2:
            print("Warning: --deepspeed requires multiple GPUs, ignoring flag")
        else:
            if importlib.util.find_spec('deepspeed') is None:
                raise RuntimeError(
                    f"deepspeed is missing from {sys.executable}. Install it in the Python environment "
                    "used to launch torchrun (python -m pip install deepspeed)."
                )
            try:
                from accelerate import DeepSpeedPlugin
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=2,
                    offload_optimizer_device="cpu" if args.cpu_offload else None,
                    zero3_init_flag=False,
                )
            except ImportError:
                print("Warning: DeepSpeed not installed, ignoring --deepspeed flag")
    elif args.cpu_offload:
        print("Warning: --cpu-offload requires --deepspeed, ignoring flag")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision or 'no',
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        log_with=None,
    )
    use_deepspeed = accelerator.state.deepspeed_plugin is not None
    if use_deepspeed and accelerator.is_main_process:
        print("\n✓ DeepSpeed ZeRO-2 enabled")
        if args.cpu_offload:
            print("✓ Optimizer state offloaded to CPU")
        print("⚠️  ZeRO shards optimizer state across ranks; checkpoints store model weights only,\n"
              "   so resuming restarts the optimizer state")
    return accelerator, use_deepspeed


def build_optimizer(model, args, use_deepspeed: bool, is_main: bool):
    """AdamW (optionally 8-bit via bitsandbytes) with betas (0.9, 0.95)."""
    kwargs = dict(lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if args.use_8bit_optimizer and not use_deepspeed:
        try:
            import bitsandbytes as bnb
            if is_main:
                print("✓ Using 8-bit AdamW optimizer (saves ~50% optimizer memory)")
            return bnb.optim.AdamW8bit(model.parameters(), **kwargs)
        except ImportError:
            if is_main:
                print("⚠️  bitsandbytes not installed, falling back to standard AdamW")
    return torch.optim.AdamW(model.parameters(), **kwargs)


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup, then cosine decay to 10% of peak, over optimizer steps.

    Step it once per real optimizer update (not through accelerator.prepare,
    which would step it once per process).
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return 0.1
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def report_schedule(steps_per_epoch: int, accum: int, epochs: int, warmup_steps: int) -> int:
    """Print the optimizer-step budget and return total optimizer steps."""
    optimizer_steps_per_epoch = steps_per_epoch // accum
    total_steps = optimizer_steps_per_epoch * epochs
    print(f"\nScheduler: {optimizer_steps_per_epoch} optimizer steps/epoch × "
          f"{epochs} epochs = {total_steps} total optimizer steps")
    if warmup_steps > 0 and total_steps > 0:
        warmup_pct = warmup_steps / total_steps * 100
        print(f"Warmup: {warmup_steps} optimizer steps ({warmup_pct:.1f}% of training)")
        if warmup_pct > 30:
            print(f"⚠️  WARNING: Warmup consumes {warmup_pct:.0f}% of training! "
                  f"Consider --warmup-steps ~{max(1, total_steps // 10)} (10% of {total_steps})")
    return total_steps


def round_steps_to_accumulation(steps_per_epoch: int, accum: int, is_main: bool) -> int:
    """Round up so no partial accumulation window leaks across epoch boundaries."""
    if accum > 1 and steps_per_epoch % accum != 0:
        rounded = ((steps_per_epoch + accum - 1) // accum) * accum
        if is_main:
            print(f"⚠️  Rounded steps_per_epoch {steps_per_epoch} → {rounded} "
                  f"(multiple of gradient_accumulation_steps={accum})")
        return rounded
    return steps_per_epoch
