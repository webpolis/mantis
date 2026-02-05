"""
Improved Training Script for HMST

Features:
- Proper HuggingFace BPE tokenizer (50K vocab)
- Validation metrics (loss, perplexity)
- Early stopping
- Learning rate scheduling
- Better logging
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import math
from hmst.models import BaseMoEModel
from hmst.configs.model_config import get_tiny_config
from hmst.tokenizer import HMSTTokenizer
from tqdm import tqdm


class TextDataset(Dataset):
    """Dataset for language modeling with proper tokenization."""

    def __init__(self, file_path, tokenizer, seq_len=512, stride=256):
        """
        Create dataset from text file.

        Args:
            file_path: Path to text file
            tokenizer: HMSTTokenizer instance
            seq_len: Sequence length
            stride: Stride for creating overlapping sequences (helps with learning)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        # Read and tokenize entire file
        print(f"Loading and tokenizing {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize in chunks to avoid memory issues
        chunk_size = 100000  # Characters per chunk
        all_tokens = []

        for i in tqdm(range(0, len(text), chunk_size), desc="Tokenizing"):
            chunk = text[i:i + chunk_size]
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            all_tokens.extend(tokens)

        self.tokens = all_tokens
        print(f"Total tokens: {len(self.tokens):,}")

        # Create sequences with stride
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            self.sequences.append((i, i + seq_len))

        print(f"Created {len(self.sequences):,} training sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        sequence = self.tokens[start:end]

        # Input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels}


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Cap at 100 to avoid overflow


def validate(model, dataloader, device):
    """
    Run validation and compute metrics.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device

    Returns:
        avg_loss: Average validation loss
        perplexity: Perplexity metric
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            output = model(input_ids)
            logits = output['logits']

            # Compute loss
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


def train_model(
    train_file,
    val_file=None,
    output_dir='./checkpoints/improved_train',
    epochs=20,
    batch_size=8,
    learning_rate=3e-4,
    seq_len=512,
    warmup_steps=1000,
    eval_every=500,
    patience=5,
    device=None
):
    """
    Train HMST model with proper setup.

    Args:
        train_file: Path to training text file
        val_file: Path to validation text file (optional)
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Peak learning rate
        seq_len: Sequence length
        warmup_steps: Linear warmup steps
        eval_every: Evaluate every N steps
        patience: Early stopping patience (epochs without improvement)
        device: Device ('cuda' or 'cpu')
    """

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    global tokenizer
    tokenizer = HMSTTokenizer()
    print(f"Vocabulary size: {len(tokenizer):,}")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = TextDataset(train_file, tokenizer, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = None
    if val_file:
        print("\nCreating validation dataset...")
        val_dataset = TextDataset(val_file, tokenizer, seq_len=seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device == 'cuda' else False
        )

    # Initialize model
    print("\nInitializing model...")
    config = get_tiny_config()

    # Update config with proper vocab size
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
    print(f"Model: {param_counts['total'] / 1e6:.2f}M total parameters, "
          f"{param_counts['active'] / 1e6:.2f}M active")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler (linear warmup + cosine decay)
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer')
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved: {tokenizer_path}")

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting training: {epochs} epochs, {len(train_loader)} batches per epoch")
    print(f"{'='*80}\n")

    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            output = model(input_ids)
            logits = output['logits']
            load_balance_loss = output.get('load_balance_loss', 0.0)

            # Compute loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            total_loss = lm_loss + load_balance_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            global_step += 1

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{lm_loss.item():.4f}',
                'ppl': f'{compute_perplexity(lm_loss.item()):.2f}',
                'lr': f'{current_lr:.2e}'
            })

            # Validation
            if val_loader and global_step % eval_every == 0:
                val_loss, val_ppl = validate(model, val_loader, device)
                print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                model.train()

        # Epoch summary
        avg_loss = epoch_lm_loss / len(train_loader)
        avg_ppl = compute_perplexity(avg_loss)

        print(f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train PPL: {avg_ppl:.2f}")

        # Validation at end of epoch
        if val_loader:
            val_loss, val_ppl = validate(model, val_loader, device)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                # Save best model
                best_path = os.path.join(output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl
                }, best_path)
                print(f"✓ New best model saved: {best_path}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered (patience={patience})")
                    break

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_loss': avg_loss
    }, final_path)

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Final model saved: {final_path}")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation perplexity: {compute_perplexity(best_val_loss):.2f}")
    print(f"{'='*80}\n")

    return model, tokenizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train HMST with improved setup')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training text file')
    parser.add_argument('--val-file', type=str, default=None,
                        help='Path to validation text file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/improved_train',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')

    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        patience=args.patience
    )
