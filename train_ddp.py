"""
Multi-GPU Training with DistributedDataParallel

Uses both GPUs efficiently with PyTorch DDP.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from hmst.models import BaseMoEModel
from hmst.configs.model_config import get_small_config
from tqdm import tqdm


class SimpleTokenizer:
    """Basic whitespace tokenizer with vocabulary building."""

    def __init__(self, vocab_size=128000):
        self.vocab_size = vocab_size
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.next_id = 4

    def build_vocab(self, text):
        """Build vocabulary from text."""
        tokens = text.lower().split()
        unique_tokens = set(tokens)

        for token in unique_tokens:
            if token not in self.token_to_id and self.next_id < self.vocab_size:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1

    def encode(self, text):
        """Convert text to token IDs."""
        tokens = text.lower().split()
        return [self.token_to_id.get(token, 1) for token in tokens]

    def decode(self, ids):
        """Convert token IDs to text."""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)


class TextDataset(Dataset):
    """Dataset that creates training sequences from text."""

    def __init__(self, text, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        self.num_sequences = max(1, (len(self.tokens) - 1) // seq_len)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        sequence = self.tokens[start:end]

        # Pad sequence if needed (only happens at end of dataset)
        # Padded positions (token_id=0) will be ignored in loss via ignore_index
        if len(sequence) < self.seq_len:
            sequence = sequence + [0] * (self.seq_len - len(sequence))

        # Next-token prediction: input is sequence[:-1], target is sequence[1:]
        # This ensures input[i] predicts labels[i] for all i
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels}


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, text_file, output_dir, epochs, batch_size, seq_len, learning_rate):
    """
    DDP training function (runs on each GPU).

    Args:
        rank: GPU rank (0 or 1)
        world_size: Total number of GPUs
        text_file: Path to training text
        output_dir: Checkpoint directory
        epochs: Number of epochs
        batch_size: Batch size per GPU
        seq_len: Sequence length
        learning_rate: Learning rate
    """
    # Setup DDP
    setup_ddp(rank, world_size)

    is_main = (rank == 0)

    if is_main:
        print(f"Training on {world_size} GPUs with DDP")
        print(f"Rank {rank} using GPU: {torch.cuda.get_device_name(rank)}")

    # Load text (only rank 0 prints)
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(text)

    if is_main:
        print(f"Vocabulary: {len(tokenizer.token_to_id)} tokens")

    # Create dataset with distributed sampler
    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    if is_main:
        print(f"Dataset: {len(dataset)} sequences, {len(dataloader)} batches per GPU")

    # Initialize model
    config = get_small_config()
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
    ).cuda(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    if is_main:
        param_counts = model.module.count_parameters()
        print(f"Model: {param_counts['total'] / 1e9:.2f}B total, {param_counts['active'] / 1e9:.2f}B active")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    if is_main:
        print(f"\n{'='*80}")
        print(f"Starting DDP training: {epochs} epochs")
        print(f"{'='*80}\n")
        os.makedirs(output_dir, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Rank {rank} Epoch {epoch+1}/{epochs}", disable=not is_main)

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].cuda(rank)
            labels = batch['labels'].cuda(rank)

            # Forward
            output = model(input_ids)
            logits = output['logits']
            load_balance_loss = output.get('load_balance_loss', 0.0)

            # Loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            total_loss = lm_loss + load_balance_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

            if is_main:
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        # Epoch complete
        avg_loss = epoch_loss / len(dataloader)

        if is_main:
            print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': {
                    'token_to_id': tokenizer.token_to_id,
                    'id_to_token': tokenizer.id_to_token
                },
                'config': config,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Cleanup
    cleanup_ddp()

    if is_main:
        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DDP Multi-GPU Training')
    parser.add_argument('input_file', type=str, help='Path to text file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/ddp_train',
                        help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')

    args = parser.parse_args()

    # Get number of GPUs
    world_size = torch.cuda.device_count()

    if world_size < 2:
        print("Error: DDP requires at least 2 GPUs")
        exit(1)

    print(f"Launching DDP training on {world_size} GPUs...")

    # Spawn training processes (one per GPU)
    mp.spawn(
        train_ddp,
        args=(world_size, args.input_file, args.output_dir, args.epochs,
              args.batch_size, args.seq_len, args.learning_rate),
        nprocs=world_size,
        join=True
    )
