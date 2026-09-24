"""
Curriculum Training Script for MANTIS Evolution Simulation

Trains a BaseMoE model on partitioned evolution simulation data with:
- Per-token loss weighting via protocol markers (computed over whole worlds)
- Curriculum mixing: token shares per partition follow a schedule over training steps
- World-boundary-aware chunking (never cross-world sequences) and world-level val splits

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
from torch.utils.data import Dataset, IterableDataset, DataLoader, ConcatDataset
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
import warnings

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')

from mantis.models import BaseMoEModel
from mantis.configs.model_config import get_micro_config, get_tiny_config, get_small_config, get_base_config
from mantis.tokenizer import MANTISTokenizer
from mantis.data import iter_documents
from mantis.utils.checkpoints import compat_load, check_tokenizer, save_training_checkpoint, restore_training_state
from mantis.training.common import (
    build_accelerator, build_optimizer, warmup_cosine_schedule, report_schedule, round_steps_to_accumulation,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EvoWorldDataset(Dataset):
    """
    Map-style dataset of one partition file, chunked per world.

    Worlds are blank-line separated; chunks never cross a world boundary.
    Per-token loss weights are computed over the whole world (so a chunk that
    starts mid-block keeps its layer's weight) and zeroed where the label is
    ignored.
    """

    def __init__(self, file_path=None, tokenizer=None, seq_len=2048, worlds=None):
        """Build from `file_path`, or wrap already-chunked `worlds` (see split_by_world)."""
        self.seq_len = seq_len
        if worlds is not None:
            self.worlds = worlds
        else:
            print(f"Loading evolution dataset: {file_path}")
            self.worlds = [self._chunk_world(tokenizer, text) for text in iter_documents(file_path)]
            print(f"  Found {len(self.worlds)} worlds")
        self.sequences = [chunk for world in self.worlds for chunk in world]
        if file_path:
            total = sum(chunk[3] for chunk in self.sequences)
            print(f"  Total tokens: {total:,}")
            print(f"  Sequences: {len(self.sequences):,}")

    def _chunk_world(self, tokenizer, text):
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        weights = tokenizer.compute_loss_weights(torch.tensor(tokens)).numpy()
        pad = tokenizer.pad_token_id
        chunks = []
        for start in range(0, len(tokens) - 1, self.seq_len):
            window = tokens[start:start + self.seq_len + 1]
            n_real = len(window)
            input_ids = np.full(self.seq_len, pad, dtype=np.int16)
            labels = np.full(self.seq_len, -100, dtype=np.int16)
            loss_w = np.zeros(self.seq_len, dtype=np.float16)
            input_ids[:n_real - 1] = window[:-1]
            labels[:n_real - 1] = window[1:]
            loss_w[:n_real - 1] = weights[start:start + n_real - 1]
            chunks.append((input_ids, labels, loss_w, n_real - 1))
        return chunks

    def split_by_world(self, val_fraction, seed=42):
        """Return (train, val) datasets holding disjoint sets of whole worlds."""
        order = list(range(len(self.worlds)))
        random.Random(seed).shuffle(order)
        n_val = max(1, int(len(order) * val_fraction))
        if n_val >= len(order):
            raise ValueError(f"Cannot split {len(order)} world(s) into train and validation")
        pick = lambda idx: EvoWorldDataset(seq_len=self.seq_len, worlds=[self.worlds[i] for i in idx])
        return pick(order[n_val:]), pick(order[:n_val])

    def mean_tokens(self):
        return sum(chunk[3] for chunk in self.sequences) / max(1, len(self.sequences))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, labels, loss_w, _ = self.sequences[idx]
        return {
            'input_ids': torch.from_numpy(input_ids.astype(np.int64)),
            'labels': torch.from_numpy(labels.astype(np.int64)),
            'loss_weights': torch.from_numpy(loss_w.astype(np.float32)),
        }


class CurriculumDataset(IterableDataset):
    """
    Endless mix of named partitions whose token shares follow a schedule.

    The training loop reports progress (fraction of planned micro-steps done)
    through `progress`; each sample picks a partition so that expected token
    shares match the scheduled proportions.
    """

    DEFAULT_SCHEDULE = [
        (0.0, {'bio': 1.0, 'eco': 0.0, 'intel': 0.0}),
        (0.2, {'bio': 0.5, 'eco': 0.5, 'intel': 0.0}),
        (0.4, {'bio': 0.25, 'eco': 0.35, 'intel': 0.40}),
        (0.6, {'bio': 0.20, 'eco': 0.30, 'intel': 0.50}),
    ]

    LINEAR_SCHEDULE = [
        (0.0, {'bio': 1.0, 'eco': 0.0, 'intel': 0.0}),
        (0.33, {'bio': 0.33, 'eco': 0.34, 'intel': 0.33}),
        (0.66, {'bio': 0.20, 'eco': 0.30, 'intel': 0.50}),
    ]

    BIO_ONLY_SCHEDULE = [
        (0.0, {'bio': 1.0, 'eco': 0.0, 'intel': 0.0}),
    ]

    def __init__(self, datasets, schedule, seed=42):
        """
        Args:
            datasets: dict of partition name ('bio', 'eco', 'intel') -> EvoWorldDataset
            schedule: list of (progress_threshold, {name: token_share})
        """
        self.names = list(datasets)
        self.datasets = [datasets[n] for n in self.names]
        self.mean_tokens = [ds.mean_tokens() for ds in self.datasets]
        self.schedule = schedule
        self.seed = seed
        self._epoch = 0
        self.progress = 0.0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def token_shares(self, progress):
        """Scheduled token shares for the present partitions at `progress`."""
        shares = self.schedule[0][1]
        for threshold, s in self.schedule:
            if progress >= threshold:
                shares = s
        props = [shares.get(n, 0.0) for n in self.names]
        total = sum(props)
        if total == 0:
            return [1.0 / len(props)] * len(props)
        return [p / total for p in props]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + self._epoch * 1000 + worker_id)

        orders = []
        for ds in self.datasets:
            order = list(range(len(ds)))
            rng.shuffle(order)
            orders.append(order)
        positions = [0] * len(self.datasets)

        while True:
            shares = self.token_shares(self.progress)
            # Sample sequences in proportion to share / mean length so tokens match the shares
            weights = [s / m for s, m in zip(shares, self.mean_tokens)]
            d = rng.choices(range(len(self.datasets)), weights=weights, k=1)[0]
            if positions[d] >= len(orders[d]):
                rng.shuffle(orders[d])
                positions[d] = 0
            item = self.datasets[d][orders[d][positions[d]]]
            positions[d] += 1
            yield item


# ---------------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------------

def weighted_cross_entropy(logits, labels, loss_weights):
    """Cross-entropy weighted per token; ignored labels carry zero weight."""
    per_token_loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view_as(labels)
    weights = loss_weights * (labels != -100)
    return (per_token_loss * weights).sum() / weights.sum().clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_perplexity(loss):
    return math.exp(min(loss, 100))


@torch.no_grad()
def validate(model, dataloader, accelerator):
    """Unweighted validation loss on this rank. Returns (summed_loss, token_count)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False, disable=not accelerator.is_main_process):
        with accelerator.autocast():
            logits = model(batch['input_ids'])['logits']
        labels = batch['labels']
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    return total_loss, total_tokens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Tokenizer
    if args.tokenizer_path:
        tokenizer = MANTISTokenizer.load(args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path} ({len(tokenizer)} tokens)")
    else:
        tokenizer = MANTISTokenizer()
        print(f"Created new tokenizer ({len(tokenizer)} tokens)")

    # Model config (attention window = training sequence length)
    checkpoint = None
    if args.resume:
        checkpoint = compat_load(args.resume)
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'config' key: {args.resume}")
        check_tokenizer(checkpoint, tokenizer, args.resume)
        config = checkpoint['config']
        if args.seq_len > config.base_moe.max_seq_len:
            print(f"Extending attention window {config.base_moe.max_seq_len} → {args.seq_len}")
            config.base_moe.max_seq_len = args.seq_len
    else:
        config = {
            'micro': get_micro_config,
            'tiny': get_tiny_config,
            'small': get_small_config,
            'base': get_base_config,
        }[args.model_size]()
        config.base_moe.max_seq_len = args.seq_len
    config.base_moe.vocab_size = len(tokenizer)

    accelerator, use_deepspeed = build_accelerator(args)
    is_main = accelerator.is_main_process
    from accelerate.utils import set_seed
    set_seed(42)

    # Datasets
    if is_main:
        print("\nLoading datasets...")
    partition_files = {'bio': args.bio, 'eco': args.eco, 'intel': args.intel}
    partitions = {name: EvoWorldDataset(path, tokenizer, seq_len=args.seq_len)
                  for name, path in partition_files.items() if path}

    val_datasets = []
    if args.val_split:
        for name, ds in list(partitions.items()):
            train_ds, val_ds = ds.split_by_world(args.val_split)
            partitions[name] = train_ds
            val_datasets.append(val_ds)
            if is_main:
                print(f"  {name}: {len(train_ds.worlds)} train / {len(val_ds.worlds)} val worlds "
                      f"({len(train_ds)} / {len(val_ds)} sequences)")
    elif args.val_file:
        val_datasets.append(EvoWorldDataset(args.val_file, tokenizer, seq_len=args.seq_len))

    schedules = {
        'default': CurriculumDataset.DEFAULT_SCHEDULE,
        'linear': CurriculumDataset.LINEAR_SCHEDULE,
        'bio-only': CurriculumDataset.BIO_ONLY_SCHEDULE,
    }
    train_dataset = CurriculumDataset(partitions, schedules[args.schedule], seed=42)
    if is_main:
        print(f"\nPartitions: {', '.join(f'{n}={partition_files[n]}' for n in partitions)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    val_loader = None
    if val_datasets:
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)

    # Model
    if is_main:
        print("\nInitializing model...")
    model = BaseMoEModel.from_config(config.base_moe)
    if is_main:
        param_counts = model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, "
              f"{param_counts['active'] / 1e6:.2f}M active, window {config.base_moe.max_seq_len} tokens")

    optimizer = build_optimizer(model, args, use_deepspeed, is_main)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # An epoch is --steps-per-epoch micro-batches; curriculum and LR both follow it
    accum = args.gradient_accumulation_steps
    steps_per_epoch = round_steps_to_accumulation(args.steps_per_epoch, accum, is_main)
    total_micro_steps = steps_per_epoch * args.epochs
    total_steps = steps_per_epoch // accum * args.epochs
    if is_main:
        report_schedule(steps_per_epoch, accum, args.epochs, args.warmup_steps)
    scheduler = warmup_cosine_schedule(optimizer, args.warmup_steps, total_steps)

    if args.gradient_checkpointing:
        accelerator.unwrap_model(model).gradient_checkpointing_enable()
        if is_main:
            print("✓ Gradient checkpointing enabled")

    # Resume from checkpoint (epoch boundaries only)
    start_epoch = 0
    optimizer_step, global_step = 0, 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if checkpoint is not None:
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        state = restore_training_state(
            checkpoint,
            optimizer=None if use_deepspeed else optimizer,
            scheduler=scheduler,
            scaler=accelerator.scaler,
        )
        start_epoch = state['epoch']
        optimizer_step, global_step = state['step'], state['global_step']
        best_val_loss = state['best_val_loss']
        epochs_without_improvement = state['epochs_without_improvement']
        if is_main:
            for w in state['warnings']:
                print(f"⚠️  {w}")
            print(f"✓ Resumed after epoch {start_epoch}, optimizer step {optimizer_step}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    def save(name, epoch, **extra):
        accelerator.wait_for_everyone()
        if is_main:
            path = os.path.join(args.output_dir, name)
            save_training_checkpoint(
                path,
                model=accelerator.unwrap_model(model),
                config=config,
                tokenizer=tokenizer,
                epoch=epoch,
                epoch_step=0,
                optimizer_step=optimizer_step,
                global_step=global_step,
                optimizer=None if use_deepspeed else optimizer,
                scheduler=scheduler,
                scaler=accelerator.scaler,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                **extra,
            )
            return path

    if is_main:
        print(f"\n{'='*80}")
        print(f"Curriculum training on {accelerator.num_processes} GPU(s): "
              f"epochs {start_epoch + 1}-{args.epochs}, {steps_per_epoch} steps/epoch")
        print(f"Schedule: {args.schedule} | Partitions: {len(partitions)}")
        print(f"{'='*80}\n")

    completed_epochs = start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_weighted_loss = 0.0
        epoch_steps = 0

        # The prepared loader re-applies its own epoch to the dataset on every iteration
        train_loader.set_epoch(epoch)
        train_dataset.progress = epoch * steps_per_epoch / total_micro_steps

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=steps_per_epoch, disable=not is_main)

        for batch in pbar:
            if epoch_steps >= steps_per_epoch:
                break

            with accelerator.accumulate(model):
                output = model(batch['input_ids'])
                logits = output['logits']
                w_loss = weighted_cross_entropy(logits, batch['labels'], batch['loss_weights'])
                accelerator.backward(w_loss + output['load_balance_loss'])

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer_step += 1

            with torch.no_grad():
                raw_loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    batch['labels'].view(-1),
                    ignore_index=-100,
                )
            epoch_weighted_loss += w_loss.item()
            epoch_loss += raw_loss.item()
            epoch_steps += 1
            global_step += 1
            train_dataset.progress = (epoch * steps_per_epoch + epoch_steps) / total_micro_steps

            pbar.set_postfix({
                'w_loss': f'{w_loss.item():.4f}',
                'raw': f'{raw_loss.item():.4f}',
                'prog': f'{train_dataset.progress:.0%}',
            })

        completed_epochs = epoch + 1
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        avg_weighted = epoch_weighted_loss / epoch_steps if epoch_steps > 0 else 0
        if is_main:
            shares = ', '.join(f'{n}={s:.2f}' for n, s in
                               zip(train_dataset.names, train_dataset.token_shares(train_dataset.progress)))
            print(f"\nEpoch {epoch+1} — raw loss: {avg_loss:.4f}, weighted loss: {avg_weighted:.4f}, "
                  f"PPL: {compute_perplexity(avg_loss):.2f} | token shares: {shares}")

        stop_early = False
        if val_loader is not None:
            raw_sum, raw_tokens = validate(accelerator.unwrap_model(model), val_loader, accelerator)
            gathered_loss = accelerator.gather(torch.tensor([raw_sum], device=accelerator.device)).sum()
            gathered_tokens = accelerator.gather(torch.tensor([float(raw_tokens)], device=accelerator.device)).sum()
            val_loss = (gathered_loss / gathered_tokens).item() if gathered_tokens > 0 else float('inf')

            if is_main:
                print(f"Epoch {epoch+1} — Val Loss: {val_loss:.4f}, Val PPL: {compute_perplexity(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if save('best_model.pt', completed_epochs, val_loss=val_loss):
                    print(f"✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if args.patience and epochs_without_improvement >= args.patience:
                if is_main:
                    print(f"\nEarly stopping: No improvement for {args.patience} epochs")
                stop_early = True

        if args.save_every and completed_epochs % args.save_every == 0:
            path = save(f'epoch_{completed_epochs}.pt', completed_epochs)
            if path:
                print(f"Checkpoint saved: {path}")

        if stop_early:
            break

    final_path = save('final_model.pt', completed_epochs)
    if is_main:
        print(f"\n{'='*80}")
        print(f"Training complete! Final model: {final_path}")
        if val_loader is not None and best_val_loss < float('inf'):
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
                        help='Fraction of worlds per partition held out for validation (e.g., 0.1)')
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
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Warmup optimizer steps (default: 2000)')
    parser.add_argument('--steps-per-epoch', type=int, required=True,
                        help='Micro-batches per epoch (the curriculum stream is endless)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping (default: 1.0)')

    # Data loading
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length; also the attention window of new models (default: 2048)')

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
                        help='Offload DeepSpeed optimizer state to CPU (requires --deepspeed)')

    args = parser.parse_args()

    # Validate
    for flag, path in (('Bio partition', args.bio), ('Eco partition', args.eco),
                       ('Intel partition', args.intel), ('Validation file', args.val_file),
                       ('Tokenizer', args.tokenizer_path), ('Checkpoint', args.resume)):
        if path and not os.path.exists(path):
            print(f"Error: {flag} not found: {path}")
            return
    if args.val_split and args.val_file:
        print("Error: Cannot use both --val-split and --val-file")
        return
    if args.val_split and (args.val_split <= 0 or args.val_split >= 1):
        print(f"Error: --val-split must be between 0 and 1, got {args.val_split}")
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
