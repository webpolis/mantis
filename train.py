"""
Unified Training Script for MANTIS

Supports:
- Single-GPU and multi-GPU training (via HuggingFace Accelerate)
- MANTIS trie tokenizer (blank-line separated documents, one EOS per document)
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

Stages 2-4 take an optional JSONL file instead of text (see mantis/training/*):
   python train.py --stage 2 data/memory.jsonl --resume ckpt.pt --tokenizer-path tok/

HuggingFace datasets (no download):
   python train.py --hf-dataset roneneldan/TinyStories --hf-val-split validation --streaming --steps-per-epoch 1000

HuggingFace datasets (use only 10% of data):
   python train.py --hf-dataset wikitext --hf-config wikitext-2-raw-v1 --hf-train-split "train[:10%]" --hf-val-split validation
"""

import os
import subprocess

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Enable expandable segments to reduce CUDA memory fragmentation.
# Without this, PyTorch's caching allocator can fail to serve large contiguous
# allocations even when total free memory is sufficient (reserved-but-fragmented).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
import json
import math
import argparse
from tqdm import tqdm
import warnings
import numpy as np

# Disable TF32 for cross-architecture compatibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Suppress cuBLAS recovery warnings (these are expected when workaround is applied)
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')

# LambdaLR.__init__ calls step() to set initial LRs before any optimizer.step() has occurred.
# This is expected and harmless — suppress the PyTorch 1.1+ warning.
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')

from mantis.models import BaseMoEModel
from mantis.configs.model_config import get_micro_config, get_tiny_config, get_small_config, get_base_config
from mantis.tokenizer import MANTISTokenizer
from mantis.data import encode_documents, pack_windows, tokenize_file, num_windows, split_at_document, split_packed
from mantis.utils.checkpoints import compat_load, check_tokenizer, save_training_checkpoint, restore_training_state
from mantis.training.common import (
    build_accelerator, build_optimizer, warmup_cosine_schedule, report_schedule, round_steps_to_accumulation,
)
from mantis.training.vram_estimator import (
    estimate_training_vram, compute_optimal_batch_sizes, format_vram_summary,
)

try:
    from datasets import load_from_disk, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class TextDataset(Dataset):
    """
    Windows of seq_len + 1 tokens over a flat token array (see mantis.data).

    Build the array with mantis.data.tokenize_file (text files) or
    texts_to_tokens (HuggingFace examples).
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int, name: str = 'dataset'):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = num_windows(len(tokens), seq_len, stride)
        if self.n_windows == 0:
            raise ValueError(
                f"{name} has {len(tokens)} tokens; at least {seq_len + 1} are needed for one sequence"
            )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        window = torch.from_numpy(self.tokens[start:start + self.seq_len + 1].astype(np.int64))
        return {'input_ids': window[:-1], 'labels': window[1:]}


class PreTokenizedDataset(Dataset):
    """
    Pre-tokenized windows saved by scripts/preprocess_data.py.

    MUCH faster than tokenizing on the fly - recommended for production training.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }


class StreamingTextDataset(IterableDataset):
    """
    HuggingFace streaming dataset, packed into windows on the fly.

    Each example is one document. Training streams are reshuffled per epoch
    with a seeded buffer. Under multiple processes Accelerate dispatches
    batches from the main process, so ranks never see duplicate data.
    """

    def __init__(self, hf_stream, tokenizer, seq_len, stride, text_column, shuffle):
        self.stream = hf_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.text_column = text_column
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _texts(self):
        stream = self.stream
        if self.shuffle:
            stream = stream.shuffle(seed=42, buffer_size=10_000)
            stream.set_epoch(self.epoch)
        for example in stream:
            if self.text_column not in example:
                raise ValueError(
                    f"Text column '{self.text_column}' not found. Available columns: {list(example)}"
                )
            yield example[self.text_column]

    def __iter__(self):
        for window in pack_windows(encode_documents(self._texts(), self.tokenizer), self.seq_len, self.stride):
            window = torch.tensor(window, dtype=torch.long)
            yield {'input_ids': window[:-1], 'labels': window[1:]}


def texts_to_tokens(texts, tokenizer) -> np.ndarray:
    """Encode documents (EOS after each) into one flat uint16 array."""
    tokens = []
    for doc_tokens in tqdm(encode_documents(texts, tokenizer), total=len(texts), desc="Tokenizing"):
        tokens.extend(doc_tokens)
    return np.asarray(tokens, dtype=np.uint16)


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))


def resolve_model_config(args, tokenizer, seq_len):
    """
    Resolve model config from checkpoint or model-size preset.

    The model's attention window (max_seq_len) is set to the training sequence
    length so inference uses exactly the context the model was trained on.

    Returns (config, checkpoint_dict_or_None) so the caller can reuse
    the loaded checkpoint later instead of deserializing it twice.
    """
    checkpoint = None
    if args.resume:
        checkpoint = compat_load(args.resume)
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'config' key: {args.resume}")
        check_tokenizer(checkpoint, tokenizer, args.resume)
        config = checkpoint['config']
        if seq_len > config.base_moe.max_seq_len:
            print(f"Extending attention window {config.base_moe.max_seq_len} → {seq_len}")
            config.base_moe.max_seq_len = seq_len
    else:
        config = {
            'micro': get_micro_config,
            'tiny': get_tiny_config,
            'small': get_small_config,
            'base': get_base_config
        }[args.model_size]()
        config.base_moe.max_seq_len = seq_len
    config.base_moe.vocab_size = len(tokenizer)
    return config, checkpoint


def _read_preprocessing_config(dataset_path):
    """Read preprocessing_config.json written by preprocess_data.py (or {} if absent)."""
    config_path = os.path.join(dataset_path, 'preprocessing_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def _check_pretokenized(dataset_path, tokenizer):
    """Return (seq_len, stride) of a pretokenized dataset and verify its tokenizer."""
    prep = _read_preprocessing_config(dataset_path)
    expected = prep.get('tokenizer_fingerprint')
    if expected is not None and expected != tokenizer.fingerprint():
        raise ValueError(
            f"{dataset_path} was tokenized with a different vocabulary "
            f"(fingerprint {expected}, current {tokenizer.fingerprint()})"
        )
    seq_len = prep.get('seq_len')
    if seq_len is None:
        seq_len = len(load_from_disk(dataset_path)[0]['input_ids'])
    return seq_len, prep.get('stride', seq_len)


def load_or_create_tokenizer(tokenizer_path=None):
    """
    Load an existing tokenizer, or create the built-in one when no path is given.
    """
    if tokenizer_path:
        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = MANTISTokenizer.load(tokenizer_path)
        print(f"Loaded vocabulary: {len(tokenizer):,} tokens")
    else:
        print("Creating new tokenizer...")
        tokenizer = MANTISTokenizer()
        print(f"Vocabulary size: {len(tokenizer):,} tokens")

    return tokenizer


def build_datasets(args, tokenizer, seq_len, is_main):
    """Return (train_dataset, val_dataset_or_None) for the selected data source."""
    stride = args.stride or seq_len

    if args.hf_dataset:
        if is_main:
            print("Using HuggingFace dataset from Hub")
        load = lambda split: load_dataset(
            args.hf_dataset, name=args.hf_config, split=split, streaming=args.streaming
        )
        train_split = args.hf_train_split or "train"

        if args.streaming:
            train = StreamingTextDataset(load(train_split), tokenizer, seq_len, stride,
                                         args.hf_text_column, shuffle=True)
            val = None
            if args.hf_val_split:
                val = StreamingTextDataset(load(args.hf_val_split), tokenizer, seq_len, seq_len,
                                           args.hf_text_column, shuffle=False)
            return train, val

        train_hf = load(train_split)
        if args.hf_text_column not in train_hf.column_names:
            raise ValueError(
                f"Text column '{args.hf_text_column}' not found. Available columns: {train_hf.column_names}"
            )
        val_hf = None
        if args.hf_val_split:
            val_hf = load(args.hf_val_split)
        elif args.val_split:
            if is_main:
                print(f"Auto-splitting documents: {args.val_split*100:.0f}% for validation")
            split = train_hf.train_test_split(test_size=args.val_split, seed=42)
            train_hf, val_hf = split['train'], split['test']

        train = TextDataset(texts_to_tokens(train_hf[args.hf_text_column], tokenizer), seq_len, stride, 'train split')
        val = None
        if val_hf is not None:
            val = TextDataset(texts_to_tokens(val_hf[args.hf_text_column], tokenizer), seq_len, seq_len, 'validation split')
        return train, val

    if args.pretokenized:
        if is_main:
            print("Using pre-tokenized datasets (fast!)")
        _, stride = _check_pretokenized(args.train_file, tokenizer)
        train_hf = load_from_disk(args.train_file)
        if args.val_file:
            _check_pretokenized(args.val_file, tokenizer)
            return PreTokenizedDataset(train_hf), PreTokenizedDataset(load_from_disk(args.val_file))
        if args.val_split:
            if is_main:
                print(f"Auto-splitting windows: last {args.val_split*100:.0f}% for validation "
                      f"(overlapping windows dropped)")
            train_hf, val_hf = split_packed(train_hf, args.val_split, seq_len, stride)
            return PreTokenizedDataset(train_hf), PreTokenizedDataset(val_hf)
        return PreTokenizedDataset(train_hf), None

    if is_main:
        print("Tokenizing on the fly (consider using --pretokenized for faster training)")
    tokens, doc_ends = tokenize_file(args.train_file, tokenizer)
    if is_main:
        print(f"Total tokens: {len(tokens):,} in {len(doc_ends):,} documents")
    if args.val_split:
        if is_main:
            print(f"Auto-splitting documents: last {args.val_split*100:.0f}% for validation")
        train_tokens, val_tokens = split_at_document(tokens, doc_ends, args.val_split)
        return (TextDataset(train_tokens, seq_len, stride, 'train split'),
                TextDataset(val_tokens, seq_len, seq_len, 'validation split'))
    val = None
    if args.val_file:
        val = TextDataset(tokenize_file(args.val_file, tokenizer)[0], seq_len, seq_len, args.val_file)
    return TextDataset(tokens, seq_len, stride, args.train_file), val


@torch.no_grad()
def validate(model, dataloader, accelerator, max_batches=None):
    """Run validation on this rank. Returns (summed_loss, token_count)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(dataloader, desc="Validating", leave=False, disable=not accelerator.is_main_process,
                    total=max_batches)

    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        with accelerator.autocast():
            logits = model(batch['input_ids'])['logits']
        labels = batch['labels']
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    return total_loss, total_tokens


def gather_val_loss(accelerator, raw_loss, raw_tokens):
    """Token-weighted validation loss across all ranks."""
    gathered_loss = accelerator.gather(torch.tensor([raw_loss], device=accelerator.device)).sum()
    gathered_tokens = accelerator.gather(torch.tensor([float(raw_tokens)], device=accelerator.device)).sum()
    return (gathered_loss / gathered_tokens).item() if gathered_tokens > 0 else float('inf')


def train(args):
    tokenizer = load_or_create_tokenizer(args.tokenizer_path)

    # Sequence length: pretokenized data fixes it at preprocessing time
    seq_len = args.seq_len
    if args.pretokenized:
        seq_len, _ = _check_pretokenized(args.train_file, tokenizer)
        if seq_len != args.seq_len:
            print(f"⚠️  Pretokenized data has seq_len={seq_len}, overriding --seq-len={args.seq_len}")

    config, preloaded_checkpoint = resolve_model_config(args, tokenizer, seq_len)

    accelerator, use_deepspeed = build_accelerator(args)
    is_main = accelerator.is_main_process
    from accelerate.utils import set_seed
    set_seed(42)

    # VRAM-aware batch size adjustment
    should_auto_batch = torch.cuda.is_available() and (accelerator.num_processes > 1 or args.auto_batch)
    if should_auto_batch:
        device_id = accelerator.local_process_index
        props = torch.cuda.get_device_properties(device_id)
        gpu_vram = props.total_memory
        gpu_name = props.name
        vram_kwargs = dict(
            mixed_precision=args.mixed_precision,
            gradient_checkpointing=args.gradient_checkpointing,
            use_8bit_optimizer=args.use_8bit_optimizer,
            deepspeed_zero_stage=2 if use_deepspeed else 0,
            num_gpus=accelerator.num_processes,
        )
        optimal_bs = compute_optimal_batch_sizes(
            config.base_moe, seq_len, [gpu_vram], safety_margin=0.85,
            max_batch_size=args.batch_size, **vram_kwargs,
        )[0]
        if is_main:
            est = estimate_training_vram(config.base_moe, seq_len, batch_size=1, **vram_kwargs)
            print(f"\n{format_vram_summary(config.base_moe, seq_len, [(gpu_name, gpu_vram)], [optimal_bs], est)}")
        print(f"[Rank {accelerator.process_index}] GPU {device_id} ({gpu_name}, "
              f"{gpu_vram / (1024**3):.1f} GB): batch_size {args.batch_size} → {optimal_bs}")
        args.batch_size = optimal_bs

    if is_main:
        print("\nLoading datasets...")
    train_dataset, val_dataset = build_datasets(args, tokenizer, seq_len, is_main)
    if is_main and not isinstance(train_dataset, IterableDataset):
        print(f"Train sequences: {len(train_dataset):,}")
        if val_dataset is not None:
            print(f"Val sequences: {len(val_dataset):,}")

    streaming = isinstance(train_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not streaming,
        num_workers=0 if streaming else args.num_workers,
        pin_memory=True
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0 if isinstance(val_dataset, IterableDataset) else args.num_workers,
            pin_memory=True
        )

    if is_main:
        print("\nInitializing model...")
    model = BaseMoEModel.from_config(config.base_moe)
    if is_main:
        param_counts = model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M total, {param_counts['active'] / 1e6:.2f}M active, "
              f"window {config.base_moe.max_seq_len} tokens")

    optimizer = build_optimizer(model, args, use_deepspeed, is_main)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # Steps per epoch from the PREPARED DataLoader. With heterogeneous GPUs and
    # per-rank batch sizes each rank yields a different number of batches; take
    # the minimum so every rank runs the same number of steps.
    accum = args.gradient_accumulation_steps
    steps_per_epoch = round_steps_to_accumulation(args.steps_per_epoch, accum, is_main) if args.steps_per_epoch else None
    if not streaming:
        # Every rank must run the same whole accumulation windows: a rank that
        # reaches the end of its loader mid-window syncs alone and hangs the others.
        steps = torch.tensor([len(train_loader)], dtype=torch.long, device=accelerator.device)
        loader_steps = int(accelerator.gather(steps).min().item()) // accum * accum
        if loader_steps == 0:
            raise ValueError(f"Fewer batches per rank than --gradient-accumulation-steps ({accum})")
        if steps_per_epoch is None or steps_per_epoch > loader_steps:
            steps_per_epoch = loader_steps

    total_steps = steps_per_epoch // accum * args.epochs
    if is_main:
        report_schedule(steps_per_epoch, accum, args.epochs, args.warmup_steps)
    scheduler = warmup_cosine_schedule(optimizer, args.warmup_steps, total_steps)

    if args.gradient_checkpointing:
        accelerator.unwrap_model(model).gradient_checkpointing_enable()
        if is_main:
            print("✓ Gradient checkpointing enabled (trades compute for memory)")

    # Resume from checkpoint (AFTER accelerator.prepare())
    start_epoch, start_epoch_step = 0, 0
    optimizer_step, global_step = 0, 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = preloaded_checkpoint
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        state = restore_training_state(
            checkpoint,
            optimizer=None if use_deepspeed else optimizer,
            scheduler=scheduler,
            scaler=accelerator.scaler,
        )
        start_epoch, start_epoch_step = state['epoch'], state['epoch_step']
        optimizer_step, global_step = state['step'], state['global_step']
        best_val_loss = state['best_val_loss']
        epochs_without_improvement = state['epochs_without_improvement']
        if is_main:
            for w in state['warnings']:
                print(f"⚠️  {w}")
            print(f"✓ Resumed at epoch {start_epoch + 1}, batch {start_epoch_step}, "
                  f"optimizer step {optimizer_step}")
            if best_val_loss < float('inf'):
                print(f"✓ Best validation loss: {best_val_loss:.4f}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer')
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved: {tokenizer_save_path}")

    def save(name, epoch, epoch_step, **extra):
        """Save on the main process after all ranks reach this point."""
        if epoch_step >= steps_per_epoch:
            epoch, epoch_step = epoch + 1, 0
        accelerator.wait_for_everyone()
        if is_main:
            path = os.path.join(args.output_dir, name)
            save_training_checkpoint(
                path,
                model=accelerator.unwrap_model(model),
                config=config,
                tokenizer=tokenizer,
                epoch=epoch,
                epoch_step=epoch_step,
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

    def run_validation():
        raw_loss, raw_tokens = validate(accelerator.unwrap_model(model), val_loader, accelerator,
                                        args.val_max_batches)
        model.train()
        return gather_val_loss(accelerator, raw_loss, raw_tokens)

    if is_main:
        print(f"\n{'='*80}")
        gpu_desc = f"{accelerator.num_processes} GPU(s)" + (" with DeepSpeed ZeRO-2" if use_deepspeed else "")
        print(f"Training on {gpu_desc}: epochs {start_epoch + 1}-{args.epochs}, "
              f"{steps_per_epoch} steps/epoch")
        print(f"{'='*80}\n")

    completed_epochs = start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # The prepared loader re-applies its own epoch to the sampler/dataset on
        # every iteration, so the epoch must be set here (the skip loader inherits it)
        train_loader.set_epoch(epoch)

        skip = start_epoch_step if epoch == start_epoch else 0
        loader = accelerator.skip_first_batches(train_loader, skip) if skip else train_loader
        epoch_loss = 0.0
        epoch_steps = skip

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", initial=skip,
                    total=steps_per_epoch, disable=not is_main)

        for batch in pbar:
            if epoch_steps >= steps_per_epoch:
                break

            with accelerator.accumulate(model):
                output = model(batch['input_ids'])
                logits = output['logits']
                lm_loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    batch['labels'].view(-1),
                    ignore_index=-100
                )
                accelerator.backward(lm_loss + output['load_balance_loss'])

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            stepped = accelerator.sync_gradients
            if stepped:
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer_step += 1

            epoch_loss += lm_loss.item()
            epoch_steps += 1
            global_step += 1
            pbar.set_postfix({'loss': f'{lm_loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

            if stepped and val_loader is not None and args.eval_every and optimizer_step % args.eval_every == 0:
                val_loss = run_validation()
                if is_main:
                    print(f"\nStep {optimizer_step} - Val Loss: {val_loss:.4f}, "
                          f"Val PPL: {compute_perplexity(val_loss):.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save('best_model.pt', epoch, epoch_steps, val_loss=val_loss):
                        print("✓ Best model saved")

        completed_epochs = epoch + 1
        trained = epoch_steps - skip
        avg_loss = epoch_loss / trained if trained else float('nan')
        if is_main:
            print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, PPL: {compute_perplexity(avg_loss):.2f}")

        stop_early = False
        if val_loader is not None:
            val_loss = run_validation()
            if is_main:
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val PPL: {compute_perplexity(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if save('best_model.pt', completed_epochs, 0, val_loss=val_loss):
                    print("✓ Best model saved")
            else:
                epochs_without_improvement += 1

            if args.patience and epochs_without_improvement >= args.patience:
                if is_main:
                    print(f"\nEarly stopping: No improvement for {args.patience} epochs")
                stop_early = True

        if args.save_every and completed_epochs % args.save_every == 0:
            path = save(f'epoch_{completed_epochs}.pt', completed_epochs, 0)
            if path:
                print(f"Checkpoint saved: {path}")

        if stop_early:
            break

    final_path = save('final_model.pt', completed_epochs, 0)
    if is_main:
        print(f"\n{'='*80}")
        print(f"Training complete! Final model: {final_path}")
        if val_loader is not None and best_val_loss < float('inf'):
            print(f"Best validation loss: {best_val_loss:.4f} (PPL: {compute_perplexity(best_val_loss):.2f})")
        print(f"{'='*80}\n")


STAGE1_ONLY_FLAGS = {
    'hf_dataset': None, 'pretokenized': False, 'streaming': False, 'val_file': None,
    'val_split': None, 'mixed_precision': False, 'deepspeed': False, 'cpu_offload': False,
    'use_8bit_optimizer': False, 'gradient_checkpointing': False, 'auto_batch': False,
}


def validate_stage1_args(args):
    """Return an error message for invalid Stage 1 arguments, or None."""
    if args.hf_dataset:
        if args.train_file:
            print("Warning: --hf-dataset provided, ignoring train_file argument")
        if args.pretokenized:
            return "Cannot use --pretokenized with --hf-dataset"
        if not DATASETS_AVAILABLE:
            return "HuggingFace datasets requires 'datasets' library. Install with: pip install datasets"
        if args.streaming and args.val_split:
            return "--val-split not supported with --streaming. Use --hf-val-split instead."
        if args.streaming and not args.steps_per_epoch:
            return "--streaming requires --steps-per-epoch (a stream has no length)"
        if args.val_file:
            return "Cannot use --val-file with --hf-dataset. Use --hf-val-split instead."
    else:
        if not args.train_file:
            return "Either train_file or --hf-dataset must be provided"
        if not os.path.exists(args.train_file):
            return f"Training {'dataset' if args.pretokenized else 'file'} not found: {args.train_file}"
        if args.val_file and not os.path.exists(args.val_file):
            return f"Validation {'dataset' if args.pretokenized else 'file'} not found: {args.val_file}"
        if args.streaming:
            return "--streaming only works with --hf-dataset"
        if args.hf_train_split or args.hf_val_split or args.hf_config or args.hf_text_column != 'text':
            return "HuggingFace-specific arguments require --hf-dataset"

    if args.resume:
        if not os.path.exists(args.resume):
            return f"Checkpoint not found: {args.resume}"
        if not args.tokenizer_path:
            tokenizer_dir = os.path.join(os.path.dirname(args.resume), 'tokenizer')
            hint = f"Try: --tokenizer-path {tokenizer_dir}" if os.path.exists(tokenizer_dir) \
                else "Use the tokenizer from the original training run"
            return f"--tokenizer-path required when resuming from checkpoint\n       {hint}"

    if args.val_split and args.val_file:
        return ("Cannot use both --val-split and --val-file. Choose one:\n"
                "  --val-split: Auto-split from training data (convenience mode)\n"
                "  --val-file: Use pre-split validation data (production mode)")
    if args.val_split and not 0 < args.val_split < 1:
        return f"--val-split must be between 0 and 1, got {args.val_split}"
    if args.stride and not 0 < args.stride <= args.seq_len:
        return f"--stride must be between 1 and --seq-len ({args.seq_len}), got {args.stride}"
    if args.pretokenized and not DATASETS_AVAILABLE:
        return "--pretokenized requires 'datasets' library. Install with: pip install datasets"
    return None


def validate_later_stage_args(args):
    """Stages 2-4 load a Stage 1 checkpoint and an optional JSONL file."""
    if not args.resume or not args.tokenizer_path:
        return (f"Stage {args.stage} requires the Stage 1 model and tokenizer:\n"
                "  --resume checkpoints/stage1/best_model.pt --tokenizer-path checkpoints/stage1/tokenizer")
    for path in (args.resume, args.tokenizer_path, args.train_file,
                 args.memory_checkpoint, args.critic_checkpoint):
        if path and not os.path.exists(path):
            return f"Not found: {path}"
    if args.semantic_store and not os.path.exists(f"{args.semantic_store}.meta"):
        return f"Semantic store not found: {args.semantic_store}.meta"
    used = [f"--{name.replace('_', '-')}" for name, default in STAGE1_ONLY_FLAGS.items()
            if getattr(args, name) != default]
    if args.gradient_accumulation_steps != 1:
        used.append('--gradient-accumulation-steps')
    if used:
        return f"Stage {args.stage} does not support: {', '.join(used)}"
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Train MANTIS Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # HUGGINGFACE DATASETS (recommended for public datasets)
  # ====================================================

  # Stream dataset without downloading (no storage needed!)
  python train.py --hf-dataset roneneldan/TinyStories \\
      --hf-val-split validation --val-max-batches 200 \\
      --streaming \\
      --steps-per-epoch 1000 \\
      --epochs 10

  # Use only 10% of dataset (avoids downloading full dataset)
  python train.py --hf-dataset wikitext \\
      --hf-config wikitext-2-raw-v1 \\
      --hf-train-split "train[:10%]" \\
      --hf-val-split validation

  # Auto-split validation from HuggingFace dataset (no validation split available)
  python train.py --hf-dataset c4 \\
      --hf-config en \\
      --hf-train-split "train[:1%]" \\
      --val-split 0.1

  # LOCAL FILES (documents separated by blank lines)
  # ================================================

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

  # Resume from checkpoint (continues mid-epoch if the checkpoint was saved mid-epoch)
  python train.py data/train.txt --resume checkpoints/train/best_model.pt \\
      --tokenizer-path checkpoints/train/tokenizer --val-split 0.1

  # Resume and train for more epochs (e.g., was 20, now train to 50 total)
  python train.py data/train.txt --resume checkpoints/train/final_model.pt \\
      --tokenizer-path checkpoints/train/tokenizer --epochs 50 --val-split 0.1

  # Mixed VRAM GPUs: Use gradient accumulation (effective batch: 2×4×N_GPUs)
  python train.py data/train.txt --gradient-accumulation-steps 4 --batch-size 2 --val-split 0.1

  # LATER STAGES (JSONL data file optional; demo data otherwise)
  # ============================================================
  python train.py --stage 2 data/memory.jsonl --resume ckpt/best_model.pt --tokenizer-path ckpt/tokenizer
  python train.py --stage 4 data/critic.jsonl --resume ckpt/best_model.pt --tokenizer-path ckpt/tokenizer
  python train.py --stage 3 data/qa.jsonl --resume ckpt/best_model.pt --tokenizer-path ckpt/tokenizer \\
      --memory-checkpoint ckpt/memory_system_final.pt --semantic-store ckpt/semantic_memory \\
      --critic-checkpoint ckpt/critic_best.pt
        """
    )

    # Data
    parser.add_argument('train_file', type=str, nargs='?',
                       help='Stage 1: text file (blank-line separated documents) or pre-tokenized dataset '
                            'directory. Stages 2-4: optional JSONL file.')
    parser.add_argument('--val-file', type=str,
                       help='Validation data: text file or pre-tokenized dataset directory')
    parser.add_argument('--val-split', type=float,
                       help='Auto-split validation fraction (e.g., 0.1 for 10%%), split at document '
                            'boundaries. Cannot be used with --val-file')
    parser.add_argument('--val-max-batches', type=int,
                        help='Cap validation at N batches per run (needed for large streamed splits)')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/train',
                        help='Output directory (default: ./checkpoints/train)')
    parser.add_argument('--pretokenized', action='store_true',
                        help='Use pre-tokenized datasets (MUCH faster, recommended for large-scale training)')

    # HuggingFace datasets
    parser.add_argument('--hf-dataset', type=str,
                        help='HuggingFace dataset name (e.g., "roneneldan/TinyStories"). '
                             'Each example is one document.')
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
                        help='Stream dataset without downloading (requires --steps-per-epoch).')

    # Tokenizer
    parser.add_argument('--tokenizer-path', type=str,
                        help='Path to existing tokenizer directory. If not provided, uses the built-in vocabulary')

    # Resumption
    parser.add_argument('--resume', type=str,
                        help='Stage 1: resume training from checkpoint. Stages 2-4: the Stage 1 model')

    # Model
    parser.add_argument('--model-size', type=str, choices=['micro', 'tiny', 'small', 'base'], default='tiny',
                        help='Model size: micro (10M), tiny (100M), small (1B), base (12B) (default: tiny)')

    # Training Stage
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], default=1,
                        help='1=Base MoE pre-training (default), 2=Memory fine-tuning, '
                             '3=RL meta-controller training, 4=Critic training')

    # Stage 3 (RL) specific options
    parser.add_argument('--rl-episodes', type=int, default=50000,
                        help='Number of RL episodes for stage 3 (default: 50000)')
    parser.add_argument('--rl-batch-size', type=int, default=256,
                        help='Episodes collected per PPO update in stage 3 (default: 256)')
    parser.add_argument('--memory-checkpoint', type=str,
                        help='Stage 3: Stage 2 output (memory_system_*.pt) enabling episodic memory')
    parser.add_argument('--semantic-store', type=str,
                        help='Stage 3: semantic memory store prefix saved by Stage 2 (e.g. ckpt/semantic_memory)')
    parser.add_argument('--critic-checkpoint', type=str,
                        help='Stage 3: Stage 4 output (critic_*.pt) enabling verification')

    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU (default: 8)')
    parser.add_argument('--learning-rate', type=float,
                        help='Peak learning rate (default: 3e-4 for stage 1, config values for stages 2-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup optimizer steps (default: 1000)')
    parser.add_argument('--steps-per-epoch', type=int, help='Max steps per epoch (default: full dataset)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping (default: 1.0)')

    # Data loading
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length; also the model attention window for new models (default: 512)')
    parser.add_argument('--stride', type=int, help='Stride for sequences (default: seq-len, non-overlapping)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')

    # Validation
    parser.add_argument('--eval-every', type=int, help='Evaluate every N optimizer steps (default: off)')
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
                        help='Enable DeepSpeed ZeRO-2 for multi-GPU training')
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Offload DeepSpeed optimizer state to CPU (requires --deepspeed)')
    parser.add_argument('--auto-batch', action='store_true',
                        help='Automatically set batch size based on VRAM estimation '
                             '(always active for multi-GPU; this flag enables it for single-GPU too). '
                             'The --batch-size value becomes the ceiling.')

    args = parser.parse_args()

    error = validate_stage1_args(args) if args.stage == 1 else validate_later_stage_args(args)
    if error:
        print(f"Error: {error}")
        return

    if args.stage == 1 and args.learning_rate is None:
        args.learning_rate = 3e-4

    if args.streaming and args.num_workers > 0:
        print(f"\n⚠️  Streaming mode detected: setting num_workers=0 (was {args.num_workers})")
        args.num_workers = 0

    # GPU configuration for Accelerate
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))

    titles = {1: "Base MoE Pre-training", 2: "Memory Fine-tuning",
              3: "RL Training (Meta-Controller Optimization)", 4: "Critic Training"}
    print(f"\n{'='*80}")
    print(f"STAGE {args.stage}: {titles[args.stage]}")
    print(f"{'='*80}\n")

    if args.stage == 1:
        train(args)
    elif args.stage == 2:
        from mantis.training.memory_train import train_memory_stage
        train_memory_stage(args)
    elif args.stage == 3:
        from mantis.training.rl_train import train_rl_stage
        train_rl_stage(args)
    else:
        from mantis.training.critic_train import train_critic_stage
        train_critic_stage(args)


if __name__ == '__main__':
    main()
