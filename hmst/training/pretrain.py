"""
Stage 1: Base MoE Pre-Training

Train the foundational model on general language understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from typing import Dict, Optional
from tqdm import tqdm
import wandb


class PreTrainer:
    """
    Pre-training manager for Stage 1.

    Trains base MoE model with:
    - Next-token prediction
    - Load balancing loss
    - Mixed-precision training
    - Gradient checkpointing

    Args:
        model: The MoE model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        learning_rate: Initial learning rate (default: 3e-4)
        weight_decay: AdamW weight decay (default: 0.01)
        warmup_steps: Number of warmup steps (default: 2000)
        total_steps: Total training steps (default: 100000)
        steps_per_epoch: Maximum steps per epoch. If None, uses full dataset (default: None)
        gradient_clip: Gradient clipping value (default: 1.0)
        checkpoint_dir: Directory for saving checkpoints (default: './checkpoints/pretrain')
        log_interval: Steps between logging (default: 100)
        save_interval: Steps between checkpoints (default: 10000)
        use_wandb: Enable W&B logging (default: False)
        device: Training device (default: 'cuda' if available else 'cpu')
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        steps_per_epoch: Optional[int] = None,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = './checkpoints/pretrain',
        log_interval: int = 100,
        save_interval: int = 10000,
        use_wandb: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.gradient_clip = gradient_clip

        # Logging
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    def train(self):
        """Main training loop."""
        print(f"Starting pre-training for {self.total_steps} steps")

        if self.use_wandb:
            wandb.init(project='hmst-pretrain', config={
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'warmup_steps': self.warmup_steps,
                'total_steps': self.total_steps,
                'steps_per_epoch': self.steps_per_epoch
            })

        self.model.train()

        while self.global_step < self.total_steps:
            epoch_loss = self._train_epoch()

            print(f"Epoch {self.epoch} | Step {self.global_step} | Loss: {epoch_loss:.4f}")

            # Validation
            if self.val_loader and self.epoch % 5 == 0:
                val_loss = self._validate()
                print(f"Validation Loss: {val_loss:.4f}")

            self.epoch += 1

        print("Pre-training complete!")

        # Final save
        self._save_checkpoint('final')

        if self.use_wandb:
            wandb.finish()

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        epoch_steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            if self.global_step >= self.total_steps:
                break

            if self.steps_per_epoch is not None and epoch_steps >= self.steps_per_epoch:
                break

            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            epoch_steps += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': self.global_step})

                if self.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.global_step
                    })

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint(f'step_{self.global_step}')

            self.global_step += 1

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict) -> float:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids

        # Warmup learning rate
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, self.global_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale
        else:
            self.scheduler.step()

        # Mixed precision forward
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            output = self.model(input_ids)
            logits = output['logits']
            load_balance_loss = output.get('load_balance_loss', 0.0)

            # Language modeling loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Total loss
            total_loss = lm_loss + load_balance_loss

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def _validate(self) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids

            output = self.model(input_ids)
            logits = output['logits']

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            total_loss += loss.item()
            num_batches += 1

        self.model.train()

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pt')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        print(f"Checkpoint loaded: {checkpoint_path}")


if __name__ == '__main__':
    # Example usage
    from ..models import BaseMoEModel

    # Initialize model
    model = BaseMoEModel(
        vocab_size=128000,
        d_model=2048,
        n_layers=24,
        n_heads=32,
        d_ff=8192,
        n_experts=8
    )

    # Dummy data loader
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 128000, (512,)),
                'labels': torch.randint(0, 128000, (512,))
            }

    train_loader = DataLoader(DummyDataset(), batch_size=4, shuffle=True)

    # Train
    trainer = PreTrainer(
        model=model,
        train_loader=train_loader,
        total_steps=1000,
        use_wandb=False
    )

    trainer.train()
