"""
Curriculum Training Script for MANTIS Evolution Simulation

Trains a BaseMoE model on partitioned evolution simulation data with:
- Per-token loss weighting via protocol markers (compute_loss_weights)
- Curriculum mixing: shift token-budget proportions across training
- World-boundary-aware chunking (never cross-world sequences)

Usage:
    # Single partition (bio only)
    python train_evo.py --bio data/evo_bio.txt --model-size micro --seq-len 256 \
        --batch-size 4 --steps-per-epoch 100 --epochs 5 --val-split 0.1

    # Full curriculum (3 partitions)
    python train_evo.py --bio data/evo_bio.txt --eco data/evo_eco.txt --intel data/evo_intel.txt \
        --model-size tiny --seq-len 2048 --batch-size 8 --steps-per-epoch 1000 --epochs 20 \
        --learning-rate 5e-4 --warmup-steps 2000 --mixed-precision --val-split 0.1

    # Resume from checkpoint
    python train_evo.py --bio data/evo_bio.txt --resume checkpoints/evo_train/best_model.pt \
        --tokenizer-path checkpoints/evo_train/tokenizer --steps-per-epoch 1000 --epochs 40 --val-split 0.1
"""

import os
import subprocess

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        gpu_names = result.stdout.strip().split('\n')
        problematic_gpus = ['RTX 30', 'RTX 40', 'A4000', 'A5000', 'A6000']
        detected_buggy = [name for name in gpu_names if any(gpu in name for gpu in problematic_gpus)]
        if detected_buggy:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
            os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
            print(f"⚠️  Detected Ampere GPU with known cuBLAS bug: {', '.join(detected_buggy)}")
            print(f"✓  Applied cuBLAS workaround (forces legacy cuBLAS, slight performance impact)")
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from accelerate import Accelerator
import math
import argparse
import random
from tqdm import tqdm
import warnings

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')

from mantis.models import BaseMoEModel
from mantis.configs.model_config import get_micro_config, get_tiny_config, get_small_config, get_base_config
from mantis.tokenizer import MANTISTokenizer
from mantis.utils.checkpoints import compat_load


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EvoWorldDataset(Dataset):
    """Map-style dataset that loads a single partition file, respecting world boundaries."""

    def __init__(self, file_path, tokenizer, seq_len=2048, stride=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

        print(f"Loading evolution dataset: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split on double-newline to get per-world text blocks
        world_blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        print(f"  Found {len(world_blocks)} worlds")

        # Tokenize each world and create chunks
        self.sequences = []  # list of (input_ids, labels, token_count)
        total_tokens = 0

        for block in world_blocks:
            tokens = tokenizer.encode(block, add_special_tokens=False)
            tokens.append(self.eos_id)
            total_tokens += len(tokens)

            # Create non-overlapping chunks of seq_len+1 tokens
            for start in range(0, len(tokens), self.stride):
                chunk = tokens[start:start + self.seq_len + 1]
                n_real = len(chunk)

                if n_real < 2:
                    continue

                # Right-pad if needed
                if n_real < self.seq_len + 1:
                    pad_count = self.seq_len + 1 - n_real
                    chunk = chunk + [self.pad_id] * pad_count
                else:
                    pad_count = 0

                input_ids = chunk[:-1]
                labels = chunk[1:]

                # Mask padding positions in labels
                if pad_count > 0:
                    for i in range(len(labels) - pad_count, len(labels)):
                        labels[i] = -100

                token_count = n_real - 1  # non-pad tokens in input
                self.sequences.append((input_ids, labels, token_count))

        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Sequences: {len(self.sequences):,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, labels, token_count = self.sequences[idx]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'token_count': token_count,
        }


class CurriculumDataset(IterableDataset):
    """Wraps 1-3 EvoWorldDataset instances with token-budget mixing."""

    DEFAULT_SCHEDULE = [
        (0.0,  [1.0,  0.0,  0.0]),
        (0.2,  [0.5,  0.5,  0.0]),
        (0.4,  [0.25, 0.35, 0.40]),
        (0.6,  [0.20, 0.30, 0.50]),
    ]

    LINEAR_SCHEDULE = [
        (0.0,  [1.0,  0.0,  0.0]),
        (0.33, [0.33, 0.34, 0.33]),
        (0.66, [0.20, 0.30, 0.50]),
    ]

    BIO_ONLY_SCHEDULE = [
        (0.0, [1.0, 0.0, 0.0]),
    ]

    def __init__(self, datasets, schedule, tokens_per_epoch, total_epochs, seed=42):
        """
        Args:
            datasets: list of 1-3 EvoWorldDataset instances [bio, eco, intel]
            schedule: list of (progress_threshold, [proportions]) — proportions per dataset
            tokens_per_epoch: token budget per epoch (controls when __iter__ stops)
            total_epochs: total training epochs (for computing global progress)
            seed: random seed
        """
        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.schedule = schedule
        self.tokens_per_epoch = tokens_per_epoch
        self.total_budget = tokens_per_epoch * total_epochs
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _get_proportions(self, progress):
        """Look up mix proportions from schedule for given progress."""
        props = self.schedule[0][1]
        for threshold, p in self.schedule:
            if progress >= threshold:
                props = p
            else:
                break
        # Trim to actual number of datasets, redistribute excess to first
        props = list(props[:self.n_datasets])
        total = sum(props)
        if total > 0:
            props = [p / total for p in props]
        else:
            props = [1.0 / self.n_datasets] * self.n_datasets
        return props

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + self._epoch * 1000 + worker_id)

        # Build shuffled index lists per dataset
        indices = []
        for ds in self.datasets:
            idx_list = list(range(len(ds)))
            rng.shuffle(idx_list)
            indices.append(idx_list)
        positions = [0] * self.n_datasets

        tokens_consumed = 0

        while tokens_consumed < self.tokens_per_epoch:
            global_tokens = self._epoch * self.tokens_per_epoch + tokens_consumed
            progress = global_tokens / self.total_budget if self.total_budget > 0 else 0.0
            props = self._get_proportions(progress)

            # Pick a dataset weighted by proportions
            ds_idx = rng.choices(range(self.n_datasets), weights=props, k=1)[0]

            # Get next item from that dataset
            ds = self.datasets[ds_idx]
            if positions[ds_idx] >= len(ds):
                # Reshuffle and reset
                idx_list = list(range(len(ds)))
                rng.shuffle(idx_list)
                indices[ds_idx] = idx_list
                positions[ds_idx] = 0

            item_idx = indices[ds_idx][positions[ds_idx]]
            positions[ds_idx] += 1

            item = ds[item_idx]
            tokens_consumed += item['token_count']

            yield item


# ---------------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------------

def weighted_cross_entropy(logits, input_ids, labels, tokenizer):
    """Cross-entropy with per-token weights from protocol layer markers."""
    weights = tokenizer.compute_loss_weights(input_ids)  # (B, seq_len)

    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view_as(labels)

    weighted = (per_token_loss * weights.to(logits.device)).sum()
    return weighted / weights.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_perplexity(loss):
    return math.exp(min(loss, 100))


@torch.no_grad()
def validate(model, dataloader, tokenizer, device, rank=0):
    """Run validation with weighted loss. Returns (avg_loss, perplexity, raw_loss, raw_tokens)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(dataloader, desc="Validating", leave=False, disable=(rank != 0))

    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids)
        logits = output['logits']

        # Raw (unweighted) loss for perplexity
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )

        total_loss += loss.item()
        non_pad_tokens = (labels != -100).sum().item()
        total_tokens += non_pad_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity, total_loss, total_tokens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    deepspeed_plugin = None
    if args.deepspeed:
        try:
            from accelerate import DeepSpeedPlugin
            if args.cpu_offload:
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=2, offload_optimizer_device="cpu", zero3_init_flag=False,
                )
            else:
                deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, zero3_init_flag=False)
            if torch.cuda.device_count() < 2:
                print("Warning: --deepspeed requires multiple GPUs, ignoring flag")
                deepspeed_plugin = None
        except ImportError:
            print("Warning: DeepSpeed not installed, ignoring --deepspeed flag")
            deepspeed_plugin = None
    elif args.cpu_offload:
        print("Warning: --cpu-offload requires --deepspeed, ignoring flag")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='fp16' if args.mixed_precision else 'no',
        deepspeed_plugin=deepspeed_plugin,
        log_with=None,
    )

    use_deepspeed = accelerator.state.deepspeed_plugin is not None

    # Tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = MANTISTokenizer.load(args.tokenizer_path)
        if accelerator.is_main_process:
            print(f"Loaded tokenizer from {args.tokenizer_path} ({len(tokenizer)} tokens)")
    else:
        tokenizer = MANTISTokenizer()
        if accelerator.is_main_process:
            print(f"Created new tokenizer ({len(tokenizer)} tokens)")

    # Model config
    checkpoint = None
    if args.resume:
        checkpoint = compat_load(args.resume)
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'config' key: {args.resume}")
        config = checkpoint['config']
        if accelerator.is_main_process:
            print(f"✓ Loaded model config from checkpoint")
        config.base_moe.vocab_size = len(tokenizer)
    else:
        config = {
            'micro': get_micro_config,
            'tiny': get_tiny_config,
            'small': get_small_config,
            'base': get_base_config,
        }[args.model_size]()
        config.base_moe.vocab_size = len(tokenizer)

    # Datasets
    if accelerator.is_main_process:
        print("\nLoading datasets...")

    partition_files = [args.bio]
    if args.eco:
        partition_files.append(args.eco)
    if args.intel:
        partition_files.append(args.intel)

    all_datasets = [
        EvoWorldDataset(f, tokenizer, seq_len=args.seq_len)
        for f in partition_files
    ]

    # Validation split
    val_datasets = None
    if args.val_split:
        val_datasets = []
        for i, ds in enumerate(all_datasets):
            n = len(ds)
            val_count = max(1, int(n * args.val_split))
            train_count = n - val_count

            # Split sequences (val = last val_count)
            val_ds = EvoWorldDataset.__new__(EvoWorldDataset)
            val_ds.tokenizer = tokenizer
            val_ds.seq_len = args.seq_len
            val_ds.stride = ds.stride
            val_ds.pad_id = ds.pad_id
            val_ds.eos_id = ds.eos_id
            val_ds.sequences = ds.sequences[train_count:]
            ds.sequences = ds.sequences[:train_count]

            val_datasets.append(val_ds)

            if accelerator.is_main_process:
                print(f"  Partition {i}: {train_count} train, {val_count} val sequences")
    elif args.val_file:
        val_datasets = [EvoWorldDataset(args.val_file, tokenizer, seq_len=args.seq_len)]

    # Estimate total tokens for curriculum budget
    total_train_tokens = sum(
        sum(seq[2] for seq in ds.sequences) for ds in all_datasets
    )
    # Scale by epochs to get full training budget
    curriculum_budget = total_train_tokens * args.epochs
    if accelerator.is_main_process:
        print(f"\nCurriculum budget: {curriculum_budget:,} tokens "
              f"({total_train_tokens:,} per epoch × {args.epochs} epochs)")
        print(f"Partitions: {len(all_datasets)} ({', '.join(partition_files)})")

    # Schedule
    schedules = {
        'default': CurriculumDataset.DEFAULT_SCHEDULE,
        'linear': CurriculumDataset.LINEAR_SCHEDULE,
        'bio-only': CurriculumDataset.BIO_ONLY_SCHEDULE,
    }
    schedule = schedules[args.schedule]

    train_dataset = CurriculumDataset(
        all_datasets, schedule, total_train_tokens,
        total_epochs=args.epochs, seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # IterableDataset
        pin_memory=True,
    )

    # Validation loader: combine all val partitions into one
    val_loader = None
    if val_datasets:
        from torch.utils.data import ConcatDataset
        combined_val = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            combined_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # Model
    if accelerator.is_main_process:
        print("\nInitializing model...")

    model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout,
        load_balance_weight=config.base_moe.load_balance_weight,
    )

    if accelerator.is_main_process:
        param_counts = model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, "
              f"{param_counts['active'] / 1e6:.2f}M active")

    # Optimizer
    if args.use_8bit_optimizer and not use_deepspeed:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.9, 0.95),
            )
            if accelerator.is_main_process:
                print("✓ Using 8-bit AdamW optimizer")
        except ImportError:
            if accelerator.is_main_process:
                print("⚠️  bitsandbytes not installed, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.9, 0.95),
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )

    # Prepare with Accelerate
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    # Steps per epoch (required for IterableDataset)
    accum = args.gradient_accumulation_steps
    steps_per_epoch = args.steps_per_epoch

    if accum > 1 and steps_per_epoch % accum != 0:
        old_steps = steps_per_epoch
        steps_per_epoch = ((steps_per_epoch + accum - 1) // accum) * accum
        if accelerator.is_main_process:
            print(f"⚠️  Rounded steps_per_epoch {old_steps} → {steps_per_epoch} "
                  f"(nearest multiple of gradient_accumulation_steps={accum})")
    args.steps_per_epoch = steps_per_epoch

    effective_steps_per_epoch = steps_per_epoch // accum
    total_steps = effective_steps_per_epoch * args.epochs

    if accelerator.is_main_process:
        print(f"\nScheduler: {effective_steps_per_epoch} optimizer steps/epoch × "
              f"{args.epochs} epochs = {total_steps} total optimizer steps")
        if args.warmup_steps > 0:
            warmup_pct = args.warmup_steps / total_steps * 100 if total_steps > 0 else 0
            print(f"Warmup: {args.warmup_steps} optimizer steps ({warmup_pct:.1f}% of training)")

    # LR scheduler: warmup + cosine decay to 10% of peak
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps if args.warmup_steps > 0 else 1.0
        if total_steps <= args.warmup_steps:
            return 0.1
        progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = accelerator.prepare(scheduler)

    if args.gradient_checkpointing:
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'gradient_checkpointing_enable'):
            unwrapped_model.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                print("✓ Gradient checkpointing enabled")

    # Resume from checkpoint
    start_epoch = 0
    start_step = 0
    start_global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if args.resume and checkpoint is not None:
        if accelerator.is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)
        start_global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        if accelerator.is_main_process:
            print(f"✓ Resumed from epoch {start_epoch}, step {start_step}")

    # Save tokenizer
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    # Training loop
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"Curriculum training on {accelerator.num_processes} GPU(s): "
              f"{args.epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"Schedule: {args.schedule} | Partitions: {len(all_datasets)}")
        print(f"{'='*80}\n")

    global_step = start_global_step
    optimizer_step = start_step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_weighted_loss = 0.0
        epoch_steps = 0

        # Set epoch for curriculum progress tracking
        train_dataset.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            total=steps_per_epoch,
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if epoch_steps >= steps_per_epoch:
                break

            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                labels = batch['labels']

                output = model(input_ids)
                logits = output['logits']
                load_balance_loss = output.get('load_balance_loss', 0.0)

                w_loss = weighted_cross_entropy(logits, input_ids, labels, tokenizer)
                total_loss = w_loss + load_balance_loss

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    scheduler.step()
                    optimizer_step += 1

            epoch_weighted_loss += w_loss.item()
            # Also track raw LM loss for comparison
            with torch.no_grad():
                raw_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            epoch_loss += raw_loss.item()
            epoch_steps += 1
            global_step += 1

            if accelerator.is_main_process:
                progress = epoch / args.epochs
                pbar.set_postfix({
                    'w_loss': f'{w_loss.item():.4f}',
                    'raw': f'{raw_loss.item():.4f}',
                    'prog': f'{progress:.0%}',
                })

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        avg_weighted = epoch_weighted_loss / epoch_steps if epoch_steps > 0 else 0

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} — raw loss: {avg_loss:.4f}, "
                  f"weighted loss: {avg_weighted:.4f}, "
                  f"PPL: {compute_perplexity(avg_loss):.2f}")

        # Validation
        if val_loader:
            val_loss, val_ppl, raw_loss, raw_tokens = validate(
                accelerator.unwrap_model(model), val_loader,
                tokenizer, accelerator.device, accelerator.process_index,
            )

            gathered_loss = accelerator.gather(
                torch.tensor(raw_loss, device=accelerator.device)).sum()
            gathered_tokens = accelerator.gather(
                torch.tensor(float(raw_tokens), device=accelerator.device)).sum()
            val_loss = (gathered_loss / gathered_tokens).item() if gathered_tokens > 0 else 0.0
            val_ppl = compute_perplexity(val_loss)

            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} — Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'step': optimizer_step,
                        'global_step': global_step,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': config,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'epochs_without_improvement': epochs_without_improvement,
                    }, best_path)
                    print(f"✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if args.patience and epochs_without_improvement >= args.patience:
                if accelerator.is_main_process:
                    print(f"\nEarly stopping: No improvement for {args.patience} epochs")
                break

        if args.save_every and (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'step': optimizer_step,
                    'global_step': global_step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'best_val_loss': best_val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': epoch + 1,
            'step': optimizer_step,
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'best_val_loss': best_val_loss,
        }, final_path)

        print(f"\n{'='*80}")
        print(f"Training complete! Final model: {final_path}")
        if val_loader:
            print(f"Best validation loss: {best_val_loss:.4f} "
                  f"(PPL: {compute_perplexity(best_val_loss):.2f})")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='MANTIS Evolution Curriculum Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data partitions
    parser.add_argument('--bio', type=str, required=True,
                        help='Bio partition file (PRIMORDIAL→CAMBRIAN, required)')
    parser.add_argument('--eco', type=str, default=None,
                        help='Eco partition file (→ECOSYSTEM, optional)')
    parser.add_argument('--intel', type=str, default=None,
                        help='Intel partition file (→INTELLIGENCE, optional)')

    # Curriculum
    parser.add_argument('--schedule', type=str, default='default',
                        choices=['default', 'linear', 'bio-only'],
                        help='Curriculum schedule (default: default)')

    # Validation
    parser.add_argument('--val-split', type=float,
                        help='Auto-split validation fraction (e.g., 0.1)')
    parser.add_argument('--val-file', type=str,
                        help='Validation data file')

    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints/evo_train',
                        help='Output directory (default: ./checkpoints/evo_train)')

    # Tokenizer
    parser.add_argument('--tokenizer-path', type=str,
                        help='Path to existing tokenizer directory')

    # Resumption
    parser.add_argument('--resume', type=str,
                        help='Resume from checkpoint')

    # Model
    parser.add_argument('--model-size', type=str,
                        choices=['micro', 'tiny', 'small', 'base'], default='tiny',
                        help='Model size (default: tiny)')

    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Peak learning rate (default: 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Warmup steps (default: 2000)')
    parser.add_argument('--steps-per-epoch', type=int, required=True,
                        help='Steps per epoch (required for curriculum IterableDataset)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping (default: 1.0)')

    # Data loading
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length (default: 2048)')

    # Validation frequency
    parser.add_argument('--patience', type=int, help='Early stopping patience in epochs')

    # Checkpointing
    parser.add_argument('--save-every', type=int, help='Save checkpoint every N epochs')

    # Performance
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--mixed-precision', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-8bit-optimizer', action='store_true',
                        help='Use 8-bit AdamW (requires bitsandbytes)')
    parser.add_argument('--deepspeed', action='store_true',
                        help='Enable DeepSpeed ZeRO-2')
    parser.add_argument('--cpu-offload', action='store_true',
                        help='CPU offload (requires --deepspeed)')

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.bio):
        print(f"Error: Bio partition not found: {args.bio}")
        return
    if args.eco and not os.path.exists(args.eco):
        print(f"Error: Eco partition not found: {args.eco}")
        return
    if args.intel and not os.path.exists(args.intel):
        print(f"Error: Intel partition not found: {args.intel}")
        return
    if args.val_split and args.val_file:
        print("Error: Cannot use both --val-split and --val-file")
        return
    if args.val_split and (args.val_split <= 0 or args.val_split >= 1):
        print(f"Error: --val-split must be between 0 and 1, got {args.val_split}")
        return
    if args.resume and not os.path.exists(args.resume):
        print(f"Error: Checkpoint not found: {args.resume}")
        return
    if args.resume and not args.tokenizer_path:
        checkpoint_dir = os.path.dirname(args.resume)
        tokenizer_dir = os.path.join(checkpoint_dir, 'tokenizer')
        print("Error: --tokenizer-path required when resuming")
        if os.path.exists(tokenizer_dir):
            print(f"       Try: --tokenizer-path {tokenizer_dir}")
        return

    print(f"\n{'='*80}")
    print("MANTIS Evolution Curriculum Training")
    print(f"{'='*80}\n")

    train(args)


if __name__ == '__main__':
    main()
