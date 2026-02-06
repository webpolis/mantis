"""
Unified Training Script for HMST

Supports:
- Single-GPU and multi-GPU training (via HuggingFace Accelerate)
- Proper tokenization (HuggingFace BPE or custom)
- Pre-tokenized datasets (recommended for large-scale training)
- HuggingFace datasets (direct from Hub with streaming support)
- Auto-split validation (convenience mode) or pre-split validation (production mode)
- Validation metrics and early stopping
- Steps per epoch control
- Learning rate scheduling
- Mixed precision training
- Heterogeneous GPU support (mixed architectures/VRAM)

Usage:

Local files (convenience mode):
   python train.py data/tokenized/train --pretokenized --val-split 0.1

Local files (production mode):
   python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
   python scripts/split_dataset.py  # creates train_split and val
   python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val

HuggingFace datasets (no download):
   python train.py --hf-dataset roneneldan/TinyStories --hf-val-split validation --streaming --steps-per-epoch 1000

HuggingFace datasets (use only 10% of data):
   python train.py --hf-dataset wikitext --hf-config wikitext-2-raw-v1 --hf-train-split "train[:10%]" --hf-val-split validation
"""

import os
import subprocess

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Detect and fix RTX 3060 cuBLAS bug
# RTX 3060 (and some other Ampere GPUs) have a cuBLASLt kernel bug with large matrices
# that causes CUBLAS_STATUS_NOT_INITIALIZED at specific sequence lengths
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        gpu_names = result.stdout.strip().split('\n')
        # Check for RTX 30xx series or A-series Ampere GPUs (known to have cuBLAS bug)
        # RTX 20xx (Turing) does NOT have this issue
        problematic_gpus = ['RTX 30', 'RTX 40', 'A4000', 'A5000', 'A6000']
        detected_buggy = [name for name in gpu_names if any(gpu in name for gpu in problematic_gpus)]

        if detected_buggy:
            # Apply cuBLAS workaround
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
            os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
            print(f"⚠️  Detected Ampere GPU with known cuBLAS bug: {', '.join(detected_buggy)}")
            print(f"✓  Applied cuBLAS workaround (forces legacy cuBLAS, slight performance impact)")
except Exception:
    # If nvidia-smi fails, silently continue (might be CPU-only or different setup)
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

# Disable TF32 for cross-architecture compatibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Suppress cuBLAS recovery warnings (these are expected when workaround is applied)
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')

from hmst.models import BaseMoEModel
from hmst.configs.model_config import get_micro_config, get_tiny_config, get_small_config, get_base_config
from hmst.tokenizer import HMSTTokenizer

try:
    from datasets import load_from_disk, load_dataset
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


class HuggingFaceDataset(Dataset):
    """
    Dataset that loads directly from HuggingFace Hub.

    Supports:
    - Streaming (no download required)
    - Dataset slicing (e.g., "train[:10%]" to use only 10% of data)
    - On-the-fly tokenization
    - Multiple dataset configurations

    Note: Streaming mode uses sequential iteration, not random access.
    DataLoader will automatically handle this correctly.
    """

    def __init__(self, dataset_name, split, tokenizer, seq_len=512,
                 streaming=False, config_name=None, text_column='text'):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., "roneneldan/TinyStories")
            split: Dataset split with optional slice (e.g., "train[:10%]", "validation")
            tokenizer: Tokenizer instance
            seq_len: Sequence length
            streaming: If True, streams data without downloading (default: False)
            config_name: Optional configuration name for datasets with multiple configs
            text_column: Name of the text column (default: 'text')
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library required for HuggingFace datasets. "
                "Install with: pip install datasets"
            )

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.streaming = streaming
        self.text_column = text_column

        print(f"Loading HuggingFace dataset: {dataset_name}")
        print(f"  Split: {split}")
        print(f"  Streaming: {streaming}")
        if config_name:
            print(f"  Config: {config_name}")

        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=streaming
        )

        if not streaming:
            dataset_len = len(self.dataset)
            if dataset_len == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' split '{split}' is empty. "
                    f"Check split name and slice syntax."
                )

            print(f"Loaded {dataset_len:,} examples")
            # Pre-tokenize all examples for faster training
            print("Pre-tokenizing dataset (this may take a moment)...")
            self._tokenized_data = []
            for idx in tqdm(range(dataset_len), desc="Tokenizing"):
                self._tokenized_data.append(self._tokenize_example(self.dataset[idx]))
            print(f"Created {len(self._tokenized_data):,} sequences")
        else:
            print("Streaming mode: data will be tokenized on-the-fly")
            # For streaming, create an iterator that will be consumed sequentially
            # PyTorch DataLoader will call __iter__ to get batches
            self._tokenized_data = None
            self._stream_iterator = None

    def _tokenize_example(self, example):
        """Tokenize a single example and create input/label pair."""
        # Validate text column exists
        if self.text_column not in example:
            available = list(example.keys())
            raise ValueError(
                f"Text column '{self.text_column}' not found in dataset. "
                f"Available columns: {available}"
            )

        text = example[self.text_column]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Truncate or pad to seq_len + 1 (for input and label)
        if len(tokens) < self.seq_len + 1:
            # Pad with pad token
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - len(tokens))
        else:
            # Truncate
            tokens = tokens[:self.seq_len + 1]

        # Create input_ids and labels for next-token prediction
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self):
        if self.streaming:
            # For streaming datasets, we can't know the length in advance
            # Return a large number and rely on steps_per_epoch
            return 10**9  # Effectively infinite
        return len(self._tokenized_data)

    def __getitem__(self, idx):
        if self.streaming:
            # Streaming mode: this should not be called with random access
            # Instead, DataLoader should iterate via __iter__
            # However, if called, we need to handle it gracefully
            # This will only work efficiently for sequential access
            if self._stream_iterator is None:
                self._stream_iterator = iter(self.dataset)

            # Try to get next item from iterator
            try:
                example = next(self._stream_iterator)
                return self._tokenize_example(example)
            except StopIteration:
                # Reset iterator if we've exhausted it
                self._stream_iterator = iter(self.dataset)
                example = next(self._stream_iterator)
                return self._tokenize_example(example)
        else:
            return self._tokenized_data[idx]


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
    deepspeed_plugin = None
    if args.deepspeed:
        try:
            from accelerate import DeepSpeedPlugin

            if args.cpu_offload:
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=2,
                    offload_optimizer_device="cpu",
                    zero3_init_flag=False,
                )
            else:
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=2,
                    zero3_init_flag=False,
                )
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
        log_with=None
    )

    use_deepspeed = accelerator.state.deepspeed_plugin is not None

    if use_deepspeed and accelerator.is_main_process:
        print("\n✓ DeepSpeed ZeRO-3 enabled")
        if args.cpu_offload:
            print("✓ CPU offloading enabled")

    tokenizer = load_or_create_tokenizer(args.tokenizer_path)

    # Datasets
    if accelerator.is_main_process:
        print("\nLoading datasets...")

    # HuggingFace datasets (direct from Hub)
    if args.hf_dataset:
        if accelerator.is_main_process:
            print("Using HuggingFace dataset from Hub")

        # Determine split (with optional slicing)
        train_split = args.hf_train_split or "train"

        train_dataset = HuggingFaceDataset(
            dataset_name=args.hf_dataset,
            split=train_split,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            streaming=args.streaming,
            config_name=args.hf_config,
            text_column=args.hf_text_column
        )

        # Validation dataset
        if args.hf_val_split:
            val_dataset = HuggingFaceDataset(
                dataset_name=args.hf_dataset,
                split=args.hf_val_split,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                streaming=args.streaming,
                config_name=args.hf_config,
                text_column=args.hf_text_column
            )
        elif args.val_split and not args.streaming:
            # Auto-split only works for non-streaming
            if accelerator.is_main_process:
                print(f"Auto-splitting dataset: {args.val_split*100:.0f}% for validation")
            # Split the underlying HF dataset
            split_dataset = train_dataset.dataset.train_test_split(
                test_size=args.val_split,
                seed=42
            )

            # Create new HuggingFaceDataset instances
            train_dataset.dataset = split_dataset['train']
            train_dataset._tokenized_data = []
            for idx in tqdm(range(len(train_dataset.dataset)), desc="Tokenizing train"):
                train_dataset._tokenized_data.append(train_dataset._tokenize_example(train_dataset.dataset[idx]))

            val_dataset = HuggingFaceDataset.__new__(HuggingFaceDataset)
            val_dataset.tokenizer = tokenizer
            val_dataset.seq_len = args.seq_len
            val_dataset.streaming = False
            val_dataset.text_column = args.hf_text_column
            val_dataset.dataset = split_dataset['test']
            val_dataset._tokenized_data = []
            for idx in tqdm(range(len(val_dataset.dataset)), desc="Tokenizing val"):
                val_dataset._tokenized_data.append(val_dataset._tokenize_example(val_dataset.dataset[idx]))

            if accelerator.is_main_process:
                print(f"Train examples: {len(train_dataset.dataset):,}")
                print(f"Val examples: {len(val_dataset.dataset):,}")
        else:
            val_dataset = None

        # Warn if streaming without steps_per_epoch
        if args.streaming and not args.steps_per_epoch:
            print("\n⚠️  WARNING: Streaming mode without --steps-per-epoch will run indefinitely!")
            print("   Recommend: --steps-per-epoch 1000 (or appropriate value)")

    elif args.pretokenized:
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

    # Determine shuffle setting - streaming datasets should not shuffle in DataLoader
    use_shuffle = True
    if args.hf_dataset and args.streaming:
        use_shuffle = False
        if accelerator.is_main_process:
            print("Note: Shuffle disabled for streaming mode (dataset streams in sequential order)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=use_shuffle,
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

    # Load config from checkpoint if resuming, otherwise create new config
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        checkpoint_for_config = torch.load(args.resume, map_location='cpu')
        if 'config' not in checkpoint_for_config:
            raise ValueError(f"Checkpoint missing 'config' key: {args.resume}")

        config = checkpoint_for_config['config']
        if accelerator.is_main_process:
            print(f"✓ Loaded model config from checkpoint")

        # Update vocab size in case tokenizer changed (though this would be unusual)
        config.base_moe.vocab_size = len(tokenizer)
    else:
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
        dropout=config.base_moe.dropout,
        load_balance_weight=config.base_moe.load_balance_weight
    )

    if accelerator.is_main_process:
        param_counts = model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, {param_counts['active'] / 1e6:.2f}M active")

    if args.use_8bit_optimizer and not use_deepspeed:
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

    if use_deepspeed and accelerator.is_main_process:
        print("✓ Using DeepSpeed ZeRO optimizer")

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

    # Prepare everything with Accelerate FIRST (before loading checkpoint)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    if args.gradient_checkpointing:
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'gradient_checkpointing_enable'):
            unwrapped_model.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                print("✓ Gradient checkpointing enabled (trades compute for memory)")

    # Resume from checkpoint if specified (AFTER accelerator.prepare())
    start_epoch = 0
    start_step = 0
    start_global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if args.resume:
        if accelerator.is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu')

        # Restore model state using unwrap_model to access actual model beneath Accelerate's wrapper
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state (Accelerate-wrapped optimizer can load state directly)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if accelerator.is_main_process:
                print("⚠️  Warning: No optimizer state in checkpoint, starting with fresh optimizer")

        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            if accelerator.is_main_process:
                print("⚠️  Warning: No scheduler state in checkpoint, starting with fresh scheduler")

        # Restore training state
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)
        start_global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        if accelerator.is_main_process:
            print(f"✓ Resumed from epoch {start_epoch}, step {start_step}")
            if best_val_loss < float('inf'):
                print(f"✓ Best validation loss: {best_val_loss:.4f}")

    # Save tokenizer (main process only)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    # Training loop
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        if args.resume:
            print(f"Resuming training on {accelerator.num_processes} GPU(s): epochs {start_epoch+1}-{args.epochs}")
        else:
            gpu_desc = f"{accelerator.num_processes} GPU(s)"
            if use_deepspeed:
                gpu_desc += " with DeepSpeed ZeRO-3"
            print(f"Training on {gpu_desc}: {args.epochs} epochs")
        if args.steps_per_epoch:
            print(f"Steps per epoch: {args.steps_per_epoch}")
        print(f"{'='*80}\n")

    global_step = start_global_step
    optimizer_step = start_step

    if accelerator.is_main_process:
        print("Starting training loop...")

    for epoch in range(start_epoch, args.epochs):
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

            if epoch_steps == 0 and accelerator.is_main_process:
                print("Got first batch, starting forward pass...")

            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                labels = batch['labels']

                output = model(input_ids)

                if epoch_steps == 0 and accelerator.is_main_process:
                    print("Forward pass complete, computing loss...")

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
                            'global_step': global_step,
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'config': config,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss
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
                        'step': optimizer_step,
                        'global_step': global_step,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': config,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'epochs_without_improvement': epochs_without_improvement
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
                    'step': optimizer_step,
                    'global_step': global_step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'best_val_loss': best_val_loss,
                    'epochs_without_improvement': epochs_without_improvement
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
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
            'best_val_loss': best_val_loss
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

  # HUGGINGFACE DATASETS (recommended for public datasets)
  # ====================================================

  # Stream dataset without downloading (no storage needed!)
  python train.py --hf-dataset roneneldan/TinyStories \\
      --hf-val-split validation \\
      --streaming \\
      --steps-per-epoch 1000 \\
      --epochs 10

  # Use only 10% of dataset (avoids downloading full dataset)
  python train.py --hf-dataset wikitext \\
      --hf-config wikitext-2-raw-v1 \\
      --hf-train-split "train[:10%]" \\
      --hf-val-split validation

  # Use first 1000 examples (great for testing)
  python train.py --hf-dataset openwebtext \\
      --hf-train-split "train[:1000]" \\
      --hf-val-split "train[1000:1200]"

  # Auto-split validation from HuggingFace dataset (no validation split available)
  python train.py --hf-dataset c4 \\
      --hf-config en \\
      --hf-train-split "train[:1%]" \\
      --val-split 0.1

  # LOCAL FILES (when you have your own data)
  # =========================================

  # CONVENIENCE MODE: Auto-split validation (quick iteration)
  python train.py data/tokenized/train --pretokenized --val-split 0.1

  # PRODUCTION MODE: Pre-tokenize AND pre-split (reproducible)
  python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
  python scripts/split_dataset.py  # splits into train_split and val
  python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val

  # Single/Multi-GPU training (auto-detected, raw text)
  python train.py data/train.txt --val-split 0.1

  # ADVANCED OPTIONS
  # ===============

  # Resume from checkpoint (continues training)
  python train.py data/train.txt --resume checkpoints/train/best_model.pt --val-split 0.1

  # Resume and train for more epochs (e.g., was 20, now train to 50 total)
  python train.py data/train.txt --resume checkpoints/train/final_model.pt --epochs 50 --val-split 0.1

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
    parser.add_argument('train_file', type=str, nargs='?',
                       help='Training data: text file or pre-tokenized dataset directory (not needed with --hf-dataset)')
    parser.add_argument('--val-file', type=str,
                       help='Validation data: text file or pre-tokenized dataset directory')
    parser.add_argument('--val-split', type=float,
                       help='Auto-split validation fraction (e.g., 0.1 for 10%%). '
                            'Convenience mode - cannot be used with --val-file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/train',
                        help='Output directory (default: ./checkpoints/train)')
    parser.add_argument('--pretokenized', action='store_true',
                        help='Use pre-tokenized datasets (MUCH faster, recommended for large-scale training)')

    # HuggingFace datasets
    parser.add_argument('--hf-dataset', type=str,
                        help='HuggingFace dataset name (e.g., "roneneldan/TinyStories"). '
                             'Loads directly from Hub without needing local files.')
    parser.add_argument('--hf-train-split', type=str,
                        help='Train split with optional slice (e.g., "train[:10%%]", "train[:1000]"). '
                             'Default: "train"')
    parser.add_argument('--hf-val-split', type=str,
                        help='Validation split (e.g., "validation", "test[:10%%]"). '
                             'If not provided, uses --val-split for auto-splitting.')
    parser.add_argument('--hf-config', type=str,
                        help='Dataset configuration name (for datasets with multiple configs)')
    parser.add_argument('--hf-text-column', type=str, default='text',
                        help='Name of the text column (default: "text")')
    parser.add_argument('--streaming', action='store_true',
                        help='Stream dataset without downloading (requires --steps-per-epoch). '
                             'Useful for very large datasets to avoid storage issues.')

    # Tokenizer
    parser.add_argument('--tokenizer-path', type=str,
                        help='Path to existing tokenizer (e.g., checkpoints/improved_train/tokenizer). '
                             'If not provided, creates new tokenizer')

    # Resumption
    parser.add_argument('--resume', type=str,
                        help='Resume training from checkpoint (e.g., checkpoints/train/best_model.pt)')

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
    parser.add_argument('--deepspeed', action='store_true',
                        help='Enable DeepSpeed ZeRO-3 for multi-GPU training (auto-detected, handles mixed VRAM)')
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Offload DeepSpeed optimizer and parameters to CPU (requires --deepspeed, slower but lower GPU memory)')

    args = parser.parse_args()

    # Validate data source
    if args.hf_dataset:
        # Using HuggingFace dataset
        if args.train_file:
            print("Warning: --hf-dataset provided, ignoring train_file argument")
        if args.pretokenized:
            print("Error: Cannot use --pretokenized with --hf-dataset")
            return
        if not DATASETS_AVAILABLE:
            print("Error: HuggingFace datasets requires 'datasets' library. Install with: pip install datasets")
            return
        if args.streaming and args.val_split:
            print("Error: --val-split not supported with --streaming. Use --hf-val-split instead.")
            return
        if args.val_file:
            print("Error: Cannot use --val-file with --hf-dataset. Use --hf-val-split instead.")
            return
    else:
        # Using local files
        if not args.train_file:
            print("Error: Either train_file or --hf-dataset must be provided")
            return
        if not os.path.exists(args.train_file):
            print(f"Error: Training {'dataset' if args.pretokenized else 'file'} not found: {args.train_file}")
            return
        if args.val_file and not os.path.exists(args.val_file):
            print(f"Error: Validation {'dataset' if args.pretokenized else 'file'} not found: {args.val_file}")
            return
        if args.streaming:
            print("Error: --streaming only works with --hf-dataset")
            return
        if args.hf_train_split or args.hf_val_split or args.hf_config or args.hf_text_column != 'text':
            print("Error: HuggingFace-specific arguments require --hf-dataset")
            return

    # Validate resumption arguments
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint not found: {args.resume}")
            return
        if not args.tokenizer_path:
            checkpoint_dir = os.path.dirname(args.resume)
            tokenizer_dir = os.path.join(checkpoint_dir, 'tokenizer')
            print("Error: --tokenizer-path required when resuming from checkpoint")
            if os.path.exists(tokenizer_dir):
                print(f"       Try: --tokenizer-path {tokenizer_dir}")
            else:
                print(f"       Use the tokenizer from the original training run")
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

    # Auto-adjust num_workers for streaming mode
    if args.streaming and args.num_workers > 0:
        print(f"\n⚠️  Streaming mode detected: setting num_workers=0 (was {args.num_workers})")
        print("   Streaming datasets require num_workers=0 to avoid shard distribution issues")
        args.num_workers = 0

    # GPU configuration for Accelerate
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))

    # Run training (Accelerate handles single/multi-GPU automatically)
    train(args)


if __name__ == '__main__':
    main()
