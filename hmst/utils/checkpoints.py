"""
Checkpoint management utilities.
"""

import torch
import os
from typing import Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    path: str,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer
        scheduler: Optional learning rate scheduler
        epoch: Current epoch
        step: Current step
        path: Save path
        metadata: Optional additional metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metadata:
        checkpoint['metadata'] = metadata

    # Create directory
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to

    Returns:
        Dict with checkpoint metadata
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded: {path}")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metadata': checkpoint.get('metadata', {})
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pt') or f.endswith('.pth')
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
        reverse=True
    )

    return os.path.join(checkpoint_dir, checkpoints[0])
