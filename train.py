"""
Unified Training Script for HMST

Supports:
- Single-GPU and multi-GPU training (via HuggingFace Accelerate)
- Proper tokenization (HuggingFace BPE or custom)
- Pre-tokenized datasets (recommended for large-scale training)
- Auto-split validation (convenience mode) or pre-split validation (production mode)
- Validation metrics and early stopping
- Steps per epoch control
- Learning rate scheduling
- Mixed precision training
- Heterogeneous GPU support (mixed architectures/VRAM)

Usage:

Convenience Mode (quick iteration):
   python train.py data/tokenized/train --pretokenized --val-split 0.1

Production Mode (reproducible):
   python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
   python scripts/split_dataset.py  # creates train_split and val
   python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val
"""

import os

# Fix for mixed GPU architectures (e.g., RTX 2060 Turing + RTX 3060 Ampere)
# MUST be set before any CUDA/PyTorch imports
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import math
import argparse
from pathlib import Path
from tqdm import tqdm

# Disable TF32 for cross-architecture compatibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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


def train(args):
    """Unified training with Accelerate (handles single/multi-GPU automatically)."""
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='fp16' if args.mixed_precision else 'no',
        log_with=None
    )

    # Tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer_path)

    # Datasets
    if accelerator.is_main_process:
        print("\nLoading datasets...")

    if args.pretokenized:
        if accelerator.is_main_process:
            print("Using pre-tokenized datasets (fast!)")
        train_dataset = PreTokenizedDataset(args.train_file)

        if args.val_split and not args.val_file:
            if accelerator.is_main_process:
                print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            full_dataset = train_dataset.dataset
            split = full_dataset.train_test_split(test_size=args.val_split, seed=42)

            train_dataset.dataset = split['train']
            val_dataset = PreTokenizedDataset.__new__(PreTokenizedDataset)
            val_dataset.dataset = split['test']

            if accelerator.is_main_process:
                print(f"Train sequences: {len(train_dataset):,}")
                print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = PreTokenizedDataset(args.val_file) if args.val_file else None
    else:
        if accelerator.is_main_process:
            print("Tokenizing on-the-fly (consider using --pretokenized for faster training)")
        train_dataset = TextDataset(args.train_file, tokenizer, args.seq_len, args.stride)

        if args.val_split and not args.val_file:
            if accelerator.is_main_process:
                print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            total_sequences = len(train_dataset.sequences)
            val_count = int(total_sequences * args.val_split)
            train_count = total_sequences - val_count

            val_sequences = train_dataset.sequences[-val_count:]
            train_dataset.sequences = train_dataset.sequences[:train_count]

            val_dataset = TextDataset.__new__(TextDataset)
            val_dataset.tokenizer = tokenizer
            val_dataset.seq_len = args.seq_len
            val_dataset.tokens = train_dataset.tokens
            val_dataset.sequences = val_sequences

            if accelerator.is_main_process:
                print(f"Train sequences: {len(train_dataset):,}")
                print(f"Val sequences: {len(val_dataset):,}")
        else:
            val_dataset = TextDataset(args.val_file, tokenizer, args.seq_len) if args.val_file else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Model
    if accelerator.is_main_process:
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
    )

    # Enable gradient checkpointing for large models
    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("✓ Gradient checkpointing enabled (trades compute for memory)")

    if accelerator.is_main_process:
        param_counts = model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, {param_counts['active'] / 1e6:.2f}M active")

    # Optimizer (with optional 8-bit for memory savings)
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.95)
            )
            if accelerator.is_main_process:
                print("✓ Using 8-bit AdamW optimizer (saves ~50% optimizer memory)")
        except ImportError:
            if accelerator.is_main_process:
                print("⚠️  bitsandbytes not installed, falling back to standard AdamW")
                print("   Install with: pip install bitsandbytes")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.95)
            )
    else:
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
                return 0.1
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare everything with Accelerate
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    # Save tokenizer (main process only)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    # Training loop
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"Training on {accelerator.num_processes} GPU(s): {args.epochs} epochs")
        if args.steps_per_epoch:
            print(f"Steps per epoch: {args.steps_per_epoch}")
        print(f"{'='*80}\n")

    global_step = 0
    optimizer_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            total=args.steps_per_epoch if args.steps_per_epoch else len(train_loader),
            disable=not accelerator.is_main_process
        )

        for batch in pbar:
            if args.steps_per_epoch and epoch_steps >= args.steps_per_epoch:
                break

            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                labels = batch['labels']

                output = model(input_ids)
                logits = output['logits']
                load_balance_loss = output.get('load_balance_loss', 0.0)

                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                total_loss = lm_loss + load_balance_loss

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    optimizer_step += 1

            epoch_loss += lm_loss.item()
            epoch_steps += 1
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({'loss': f'{lm_loss.item():.4f}'})

            if args.eval_every and optimizer_step > 0 and optimizer_step % args.eval_every == 0:
                if val_loader:
                    val_loss, val_ppl = validate(
                        accelerator.unwrap_model(model),
                        val_loader,
                        tokenizer,
                        accelerator.device,
                        accelerator.process_index
                    )

                    # Gather validation loss from all processes
                    val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
                    val_loss_tensor = accelerator.gather(val_loss_tensor).mean()
                    val_loss = val_loss_tensor.item()
                    val_ppl = compute_perplexity(val_loss)

                    if accelerator.is_main_process:
                        print(f"\nStep {optimizer_step} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

                    if val_loss < best_val_loss and accelerator.is_main_process:
                        best_val_loss = val_loss
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        best_path = os.path.join(args.output_dir, 'best_model.pt')
                        torch.save({
                            'epoch': epoch + 1,
                            'step': optimizer_step,
                            'model_state_dict': unwrapped_model.state_dict(),
                            'config': config,
                            'val_loss': val_loss
                        }, best_path)
                        print(f"✓ Best model saved")

                model.train()

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, PPL: {compute_perplexity(avg_loss):.2f}")

        # Validation (all processes participate)
        if val_loader:
            val_loss, val_ppl = validate(
                accelerator.unwrap_model(model),
                val_loader,
                tokenizer,
                accelerator.device,
                accelerator.process_index
            )

            # Gather validation loss from all processes
            val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
            val_loss_tensor = accelerator.gather(val_loss_tensor).mean()
            val_loss = val_loss_tensor.item()
            val_ppl = compute_perplexity(val_loss)

            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if accelerator.is_main_process:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'config': config,
                        'val_loss': val_loss
                    }, best_path)
                    print(f"✓ Best model saved")
            else:
                epochs_without_improvement += 1

            if args.patience and epochs_without_improvement >= args.patience:
                if accelerator.is_main_process:
                    print(f"\nEarly stopping: No improvement for {args.patience} epochs")
                break

        if accelerator.is_main_process:
            if args.save_every and (epoch + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'config': config
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.epochs,
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'config': config
        }, final_path)

        print(f"\n{'='*80}")
        print(f"Training complete! Final model: {final_path}")
        if val_loader:
            print(f"Best validation loss: {best_val_loss:.4f} (PPL: {compute_perplexity(best_val_loss):.2f})")
        print(f"{'='*80}\n")


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

  # Single/Multi-GPU training (auto-detected, raw text)
  python train.py data/train.txt --val-split 0.1

  # Use specific GPUs (e.g., only GPU 0)
  python train.py data/tokenized/train --pretokenized --gpu-ids 0 --val-split 0.1

  # Mixed VRAM GPUs: Use gradient accumulation (effective batch: 2×4×N_GPUs)
  python train.py data/train.txt --gradient-accumulation-steps 4 --batch-size 2 --val-split 0.1

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
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                        help='Specific GPU IDs to use (e.g., --gpu-ids 0 or --gpu-ids 0 1). '
                             'Default: uses all available GPUs')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch per GPU = batch_size × accumulation_steps. '
                             'Essential for mixed VRAM GPUs (e.g., 12GB + 6GB) (default: 1)')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision (FP16)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (trades compute for memory, essential for large models)')
    parser.add_argument('--use-8bit-optimizer', action='store_true',
                        help='Use 8-bit AdamW optimizer (saves ~50%% optimizer memory, requires bitsandbytes)')

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

    # GPU configuration for Accelerate
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))

    # Run training (Accelerate handles single/multi-GPU automatically)
    train(args)


if __name__ == '__main__':
    main()
