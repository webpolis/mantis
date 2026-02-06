"""
Unified Training Script for HMST

Supports:
- Single-GPU and multi-GPU (DDP) training
- Proper tokenization (HuggingFace BPE or custom)
- Pre-tokenized datasets (recommended for large-scale training)
- Auto-split validation (convenience mode) or pre-split validation (production mode)
- Validation metrics and early stopping
- Steps per epoch control
- Learning rate scheduling
- Mixed precision training

Usage:

Convenience Mode (quick iteration):
   python train.py data/tokenized/train --pretokenized --val-split 0.1

Production Mode (reproducible):
   python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
   python scripts/split_dataset.py  # creates train_split and val
   python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math
import argparse
from pathlib import Path
from tqdm import tqdm

from hmst.models import BaseMoEModel
from hmst.configs.model_config import get_micro_config, get_tiny_config, get_small_config, get_base_config
from hmst.tokenizer import HMSTTokenizer

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class TextDataset(Dataset):
    """Efficient dataset for language modeling."""

    def __init__(self, file_path, tokenizer, seq_len=512, stride=None):
        """
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer instance
            seq_len: Sequence length
            stride: Stride for overlapping sequences (default: seq_len for non-overlapping)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize in chunks
        chunk_size = 100000
        all_tokens = []
        for i in tqdm(range(0, len(text), chunk_size), desc="Tokenizing"):
            chunk = text[i:i + chunk_size]
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            all_tokens.extend(tokens)

        self.tokens = all_tokens
        print(f"Total tokens: {len(self.tokens):,}")

        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, self.stride):
            self.sequences.append((i, i + seq_len))

        if len(self.sequences) == 0:
            raise ValueError(
                f"Text file too small: {len(self.tokens)} tokens < {seq_len} sequence length. "
                f"File must have at least {seq_len + 1} tokens."
            )

        print(f"Created {len(self.sequences):,} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        sequence = self.tokens[start:end]

        # Next-token prediction
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels}


class PreTokenizedDataset(Dataset):
    """
    Dataset that loads pre-tokenized data from HuggingFace datasets cache.

    MUCH faster than TextDataset - recommended for production training.
    Use scripts/preprocess_data.py to create pre-tokenized datasets.
    """

    def __init__(self, dataset_path):
        """
        Args:
            dataset_path: Path to pre-tokenized dataset directory
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library required for pre-tokenized datasets. "
                "Install with: pip install datasets"
            )

        print(f"Loading pre-tokenized dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        print(f"Loaded {len(self.dataset):,} sequences")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))


def get_gpu_device_count():
    """Get number of available CUDA devices.

    Note: This initializes CUDA context. Should not be called in main process
    before mp.spawn().

    Returns:
        int: Number of CUDA devices
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_gpu_memory_info():
    """Get available VRAM for each GPU in GB.

    Note: This initializes CUDA context. Should not be called in main process
    before mp.spawn().

    Returns:
        list: List of dicts with GPU info (id, name, memory_gb)
    """
    if not torch.cuda.is_available():
        return []

    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024 ** 3)
        gpu_memory.append({
            'id': i,
            'name': props.name,
            'memory_gb': total_memory_gb
        })
    return gpu_memory


def check_gpu_homogeneity(gpu_ids=None):
    """
    Check if selected GPUs have similar VRAM and warn if heterogeneous.

    Returns:
        bool: True if GPUs are homogeneous (within 20% difference), False otherwise
    """
    gpu_info = get_gpu_memory_info()

    if gpu_ids:
        gpu_info = [gpu_info[i] for i in gpu_ids if i < len(gpu_info)]

    if len(gpu_info) < 2:
        return True

    memories = [gpu['memory_gb'] for gpu in gpu_info]
    min_mem = min(memories)
    max_mem = max(memories)

    print("\n" + "="*80)
    print("GPU Configuration:")
    for idx, gpu in enumerate(gpu_info):
        if gpu_ids and len(gpu_ids) > 1:
            print(f"  System GPU {gpu['id']} (DDP rank {idx}): {gpu['name']} - {gpu['memory_gb']:.1f} GB VRAM")
        else:
            print(f"  GPU {gpu['id']}: {gpu['name']} - {gpu['memory_gb']:.1f} GB VRAM")
    print("="*80)

    # Check if difference is significant (>20%)
    if max_mem > min_mem * 1.2:
        print("\n⚠️  WARNING: Heterogeneous GPU configuration detected!")
        print(f"   GPU memory ranges from {min_mem:.1f} GB to {max_mem:.1f} GB")
        print("\n   PyTorch DDP assumes homogeneous GPUs. Recommendations:")

        # Find the GPU with most memory
        largest_gpu = max(gpu_info, key=lambda g: g['memory_gb'])
        print(f"   1. Use only the larger GPU: --gpu-ids {largest_gpu['id']}")
        print(f"   2. Use gradient accumulation: --gradient-accumulation-steps 4 --batch-size N")
        print(f"   3. Use smaller batch size that fits on {min_mem:.1f} GB GPU")
        print("\n   Proceeding will use batch size that fits on smallest GPU.\n")
        return False

    return True


def setup_ddp(rank, world_size, device_id):
    """Initialize DDP process group.

    Args:
        rank: Process rank (0, 1, 2, ...)
        world_size: Total number of processes
        device_id: Physical GPU ID to use for this process
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device_id)


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def load_or_create_tokenizer(tokenizer_path=None):
    """
    Load existing tokenizer or create new one.

    Args:
        tokenizer_path: Path to saved tokenizer directory (e.g., '/checkpoints/tokenizer')
                       If None, creates a new tokenizer

    Returns:
        HMSTTokenizer instance
    """
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = HMSTTokenizer.load(tokenizer_path)
        print(f"Loaded vocabulary: {len(tokenizer):,} tokens")
    else:
        print("Creating new tokenizer...")
        tokenizer = HMSTTokenizer()
        print(f"Vocabulary size: {len(tokenizer):,} tokens")

    return tokenizer


@torch.no_grad()
def validate(model, dataloader, tokenizer, device, rank=0):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(dataloader, desc="Validating", leave=False, disable=(rank != 0))

    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids)
        logits = output['logits']

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


def train_single_gpu(args):
    """Single-GPU training."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer_path)

    # Datasets
    print("\nLoading datasets...")
    if args.pretokenized:
        # Fast path: load pre-tokenized data
        print("Using pre-tokenized datasets (fast!)")
        train_dataset = PreTokenizedDataset(args.train_file)

        # Auto-split if requested
        if args.val_split and not args.val_file:
            print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            full_dataset = train_dataset.dataset
            split = full_dataset.train_test_split(test_size=args.val_split, seed=42)

            # Create new datasets from splits
            train_dataset.dataset = split['train']
            val_dataset = PreTokenizedDataset.__new__(PreTokenizedDataset)
            val_dataset.dataset = split['test']

            print(f"Train sequences: {len(train_dataset):,}")
            print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = PreTokenizedDataset(args.val_file) if args.val_file else None
    else:
        # Slow path: tokenize on-the-fly
        print("Tokenizing on-the-fly (consider using --pretokenized for faster training)")
        train_dataset = TextDataset(args.train_file, tokenizer, args.seq_len, args.stride)

        # Auto-split if requested
        if args.val_split and not args.val_file:
            print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            total_sequences = len(train_dataset.sequences)
            val_count = int(total_sequences * args.val_split)
            train_count = total_sequences - val_count

            # Split sequences
            val_sequences = train_dataset.sequences[-val_count:]
            train_dataset.sequences = train_dataset.sequences[:train_count]

            # Create validation dataset (shares tokens)
            val_dataset = TextDataset.__new__(TextDataset)
            val_dataset.tokenizer = tokenizer
            val_dataset.seq_len = args.seq_len
            val_dataset.tokens = train_dataset.tokens
            val_dataset.sequences = val_sequences

            print(f"Train sequences: {len(train_dataset):,}")
            print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = TextDataset(args.val_file, tokenizer, args.seq_len) if args.val_file else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda')
        )

    # Model
    print("\nInitializing model...")
    config = {
        'micro': get_micro_config,
        'tiny': get_tiny_config,
        'small': get_small_config,
        'base': get_base_config
    }[args.model_size]()

    config.base_moe.vocab_size = len(tokenizer)

    model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Model: {param_counts['total'] / 1e6:.2f}M total, {param_counts['active'] / 1e6:.2f}M active")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler (adjusted for gradient accumulation)
    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // args.gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            if args.warmup_steps == 0:
                return 1.0
            return step / args.warmup_steps
        else:
            if total_steps <= args.warmup_steps:
                return 0.1  # Min LR if no decay phase
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device == 'cuda' else None

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
    tokenizer.save(tokenizer_save_path)
    print(f"Tokenizer saved: {tokenizer_save_path}")

    # Training loop
    print(f"\n{'='*80}")
    print(f"Training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    if args.steps_per_epoch:
        print(f"Steps per epoch limited to: {args.steps_per_epoch}")
    print(f"{'='*80}\n")

    global_step = 0
    optimizer_step = 0  # Track actual optimizer updates (not batch count)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()  # Initialize gradients at start of each epoch

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            total=args.steps_per_epoch if args.steps_per_epoch else len(train_loader)
        )

        for batch in pbar:
            if args.steps_per_epoch and epoch_steps >= args.steps_per_epoch:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                output = model(input_ids)
                logits = output['logits']
                load_balance_loss = output.get('load_balance_loss', 0.0)

                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                total_loss = lm_loss + load_balance_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / args.gradient_accumulation_steps

            # Backward
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Gradient accumulation
            if (epoch_steps + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                optimizer_step += 1

                # Mid-epoch validation (based on optimizer steps, not batch count)
                if val_loader and args.eval_every and optimizer_step % args.eval_every == 0:
                    val_loss, val_ppl = validate(model, val_loader, tokenizer, device)
                    print(f"\nOptimizer Step {optimizer_step} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                    model.train()

            epoch_loss += lm_loss.item()
            epoch_steps += 1
            global_step += 1

            # Progress
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{lm_loss.item():.4f}',
                'ppl': f'{compute_perplexity(lm_loss.item()):.1f}',
                'lr': f'{current_lr:.2e}'
            })

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps
        avg_ppl = compute_perplexity(avg_loss)
        print(f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train PPL: {avg_ppl:.2f}")

        # End-of-epoch validation
        if val_loader:
            val_loss, val_ppl = validate(model, val_loader, tokenizer, device)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                best_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl
                }, best_path)
                print(f"✓ Best model saved: {best_path}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                if args.patience and epochs_without_improvement >= args.patience:
                    print(f"\nEarly stopping (patience={args.patience})")
                    break

        # Checkpoint
        if args.save_every and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)

    print(f"\n{'='*80}")
    print(f"Training complete! Final model: {final_path}")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f} (PPL: {compute_perplexity(best_val_loss):.2f})")
    print(f"{'='*80}\n")


def train_ddp_worker(rank, world_size, args):
    """DDP worker function.

    Args:
        rank: Process rank (0, 1, 2, ...)
        world_size: Total number of processes
        args: Training arguments
    """
    # Map rank to actual GPU ID from gpu_ids list
    gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank

    setup_ddp(rank, world_size, gpu_id)
    is_main = (rank == 0)

    # Tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer_path)

    # Datasets
    if is_main:
        print("\nLoading datasets...")

    if args.pretokenized:
        # Fast path: load pre-tokenized data
        if is_main:
            print("Using pre-tokenized datasets (fast!)")
        train_dataset = PreTokenizedDataset(args.train_file)

        # Auto-split if requested
        if args.val_split and not args.val_file:
            if is_main:
                print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            full_dataset = train_dataset.dataset
            split = full_dataset.train_test_split(test_size=args.val_split, seed=42)

            # Create new datasets from splits
            train_dataset.dataset = split['train']
            val_dataset = PreTokenizedDataset.__new__(PreTokenizedDataset)
            val_dataset.dataset = split['test']

            if is_main:
                print(f"Train sequences: {len(train_dataset):,}")
                print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = PreTokenizedDataset(args.val_file) if args.val_file else None
    else:
        # Slow path: tokenize on-the-fly
        if is_main:
            print("Tokenizing on-the-fly (consider using --pretokenized for faster training)")
        train_dataset = TextDataset(args.train_file, tokenizer, args.seq_len, args.stride)

        # Auto-split if requested
        if args.val_split and not args.val_file:
            if is_main:
                print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            total_sequences = len(train_dataset.sequences)
            val_count = int(total_sequences * args.val_split)
            train_count = total_sequences - val_count

            # Split sequences
            val_sequences = train_dataset.sequences[-val_count:]
            train_dataset.sequences = train_dataset.sequences[:train_count]

            # Create validation dataset (shares tokens)
            val_dataset = TextDataset.__new__(TextDataset)
            val_dataset.tokenizer = tokenizer
            val_dataset.seq_len = args.seq_len
            val_dataset.tokens = train_dataset.tokens
            val_dataset.sequences = val_sequences

            if is_main:
                print(f"Train sequences: {len(train_dataset):,}")
                print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = TextDataset(args.val_file, tokenizer, args.seq_len) if args.val_file else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Model
    if is_main:
        print("\nInitializing model...")

    config = {
        'micro': get_micro_config,
        'tiny': get_tiny_config,
        'small': get_small_config,
        'base': get_base_config
    }[args.model_size]()

    config.base_moe.vocab_size = len(tokenizer)

    model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout
    ).cuda(gpu_id)

    model = DDP(model, device_ids=[gpu_id])

    if is_main:
        param_counts = model.module.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, {param_counts['active'] / 1e6:.2f}M active")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler (adjusted for gradient accumulation)
    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // args.gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            if args.warmup_steps == 0:
                return 1.0
            return step / args.warmup_steps
        else:
            if total_steps <= args.warmup_steps:
                return 0.1  # Min LR if no decay phase
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Save tokenizer (rank 0 only)
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    # Training
    if is_main:
        print(f"\n{'='*80}")
        print(f"DDP Training on {world_size} GPUs: {args.epochs} epochs")
        if args.steps_per_epoch:
            print(f"Steps per epoch: {args.steps_per_epoch}")
        print(f"{'='*80}\n")

    global_step = 0
    optimizer_step = 0  # Track actual optimizer updates (not batch count)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()  # Initialize gradients at start of each epoch

        pbar = tqdm(
            train_loader,
            desc=f"Rank {rank} Epoch {epoch+1}",
            total=args.steps_per_epoch if args.steps_per_epoch else len(train_loader),
            disable=not is_main
        )

        for batch in pbar:
            if args.steps_per_epoch and epoch_steps >= args.steps_per_epoch:
                break

            input_ids = batch['input_ids'].cuda(gpu_id)
            labels = batch['labels'].cuda(gpu_id)

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                output = model(input_ids)
                logits = output['logits']
                load_balance_loss = output.get('load_balance_loss', 0.0)

                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                total_loss = lm_loss + load_balance_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / args.gradient_accumulation_steps

            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Gradient accumulation
            if (epoch_steps + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                optimizer_step += 1

            epoch_loss += lm_loss.item()
            epoch_steps += 1
            global_step += 1

            if is_main:
                pbar.set_postfix({'loss': f'{lm_loss.item():.4f}'})

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps

        if is_main:
            print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, PPL: {compute_perplexity(avg_loss):.2f}")

            # Validation
            if val_loader:
                val_loss, val_ppl = validate(model, val_loader, tokenizer, f'cuda:{gpu_id}', rank)
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict(),
                        'config': config,
                        'val_loss': val_loss
                    }, best_path)
                    print(f"✓ Best model saved")

            # Checkpoint
            if args.save_every and (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'config': config
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    if is_main:
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.module.state_dict(),
            'config': config
        }, final_path)
        print(f"\n{'='*80}")
        print(f"Training complete! Final model: {final_path}")
        print(f"{'='*80}\n")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(
        description='Train HMST Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CONVENIENCE MODE: Auto-split validation (quick iteration)
  python train.py data/tokenized/train --pretokenized --val-split 0.1

  # PRODUCTION MODE: Pre-tokenize AND pre-split (reproducible)
  python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
  python scripts/split_dataset.py  # splits into train_split and val
  python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val

  # Single GPU training (raw text, slower)
  python train.py data/train.txt --val-split 0.1

  # Multi-GPU training with auto-split (all GPUs)
  python train.py data/tokenized/train --pretokenized --multi-gpu --val-split 0.1

  # Multi-GPU with specific GPUs (e.g., only GPU 0 and 2)
  python train.py data/tokenized/train --pretokenized --multi-gpu --gpu-ids 0 2 --val-split 0.1

  # Mixed VRAM GPUs: Use gradient accumulation (effective batch: 2×4×2 = 16)
  python train.py data/train.txt --multi-gpu --gradient-accumulation-steps 4 --batch-size 2 --val-split 0.1

  # Use existing tokenizer from checkpoint
  python train.py data/train.txt --tokenizer-path checkpoints/improved_train/tokenizer --val-split 0.1

  # Control steps per epoch (useful for large datasets)
  python train.py data/train.txt --steps-per-epoch 1000 --val-split 0.1

  # Train larger model
  python train.py data/train.txt --model-size small --batch-size 4 --val-split 0.1
        """
    )

    # Data
    parser.add_argument('train_file', type=str,
                       help='Training data: text file or pre-tokenized dataset directory')
    parser.add_argument('--val-file', type=str,
                       help='Validation data: text file or pre-tokenized dataset directory')
    parser.add_argument('--val-split', type=float,
                       help='Auto-split validation fraction (e.g., 0.1 for 10%%). '
                            'Convenience mode - cannot be used with --val-file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/train',
                        help='Output directory (default: ./checkpoints/train)')
    parser.add_argument('--pretokenized', action='store_true',
                        help='Use pre-tokenized datasets (MUCH faster, recommended for large-scale training)')

    # Tokenizer
    parser.add_argument('--tokenizer-path', type=str,
                        help='Path to existing tokenizer (e.g., checkpoints/improved_train/tokenizer). '
                             'If not provided, creates new tokenizer')

    # Model
    parser.add_argument('--model-size', type=str, choices=['micro', 'tiny', 'small', 'base'], default='tiny',
                        help='Model size: micro (10M), tiny (100M), small (1B), base (12B) (default: tiny)')

    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Peak learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup steps (default: 1000)')
    parser.add_argument('--steps-per-epoch', type=int, help='Max steps per epoch (default: full dataset)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping (default: 1.0)')

    # Data loading
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length (default: 512)')
    parser.add_argument('--stride', type=int, help='Stride for sequences (default: seq-len, non-overlapping)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')

    # Validation
    parser.add_argument('--eval-every', type=int, help='Evaluate every N steps (default: off)')
    parser.add_argument('--patience', type=int, help='Early stopping patience in epochs (default: off)')

    # Checkpointing
    parser.add_argument('--save-every', type=int, help='Save checkpoint every N epochs (default: off)')

    # Performance
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU DDP training')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='Specific GPU IDs to use (e.g., --gpu-ids 0 2). Default: all available GPUs')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch per GPU = batch_size × accumulation_steps. '
                             'Total effective batch = batch_size × accumulation_steps × num_gpus. '
                             'Essential for mixed VRAM GPUs (e.g., 12GB + 6GB) (default: 1)')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision (FP16)')

    args = parser.parse_args()

    # Validate inputs exist
    if not os.path.exists(args.train_file):
        print(f"Error: Training {'dataset' if args.pretokenized else 'file'} not found: {args.train_file}")
        return
    if args.val_file and not os.path.exists(args.val_file):
        print(f"Error: Validation {'dataset' if args.pretokenized else 'file'} not found: {args.val_file}")
        return

    # Validate val-split arguments
    if args.val_split and args.val_file:
        print("Error: Cannot use both --val-split and --val-file. Choose one:")
        print("  --val-split: Auto-split from training data (convenience mode)")
        print("  --val-file: Use pre-split validation data (production mode)")
        return
    if args.val_split and (args.val_split <= 0 or args.val_split >= 1):
        print(f"Error: --val-split must be between 0 and 1, got {args.val_split}")
        return

    # Validate pretokenized mode
    if args.pretokenized and not DATASETS_AVAILABLE:
        print("Error: --pretokenized requires 'datasets' library. Install with: pip install datasets")
        return

    # Multi-GPU training
    if args.multi_gpu:
        # Get GPU count WITHOUT initializing CUDA in main process
        # Use subprocess to avoid corrupting worker contexts
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        def _get_count(q):
            q.put(get_gpu_device_count())

        p = ctx.Process(target=_get_count, args=(queue,))
        p.start()
        p.join()
        available_gpus = queue.get()

        # Determine which GPUs to use
        if args.gpu_ids:
            gpu_ids = args.gpu_ids
            # Validate GPU IDs
            for gpu_id in gpu_ids:
                if gpu_id >= available_gpus:
                    print(f"Error: GPU {gpu_id} not available. Only {available_gpus} GPU(s) detected.")
                    return
            world_size = len(gpu_ids)
        else:
            world_size = available_gpus
            gpu_ids = list(range(world_size))

        # CRITICAL: Populate args.gpu_ids so workers can access it
        args.gpu_ids = gpu_ids

        if world_size < 2:
            print("Error: --multi-gpu requires at least 2 GPUs")
            return

        # Check GPU homogeneity
        # CRITICAL: Run in subprocess to avoid initializing CUDA in main process!
        # PyTorch forbids CUDA initialization before mp.spawn - it corrupts worker contexts
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=check_gpu_homogeneity, args=(gpu_ids,))
        p.start()
        p.join()

        # Warn about gradient accumulation
        if args.gradient_accumulation_steps > 1:
            effective_batch = args.batch_size * args.gradient_accumulation_steps
            print(f"\n💡 Gradient Accumulation: {args.gradient_accumulation_steps} steps")
            print(f"   Effective batch size per GPU: {effective_batch}")
            print(f"   Total effective batch size: {effective_batch * world_size}\n")

        print(f"Launching DDP training on {world_size} GPU(s): {gpu_ids}...")

        # Spawn processes - each worker will use explicit GPU ID from gpu_ids list
        mp.spawn(train_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_single_gpu(args)


if __name__ == '__main__':
    main()
