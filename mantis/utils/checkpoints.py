"""
Checkpoint management utilities.

Every training checkpoint carries its model config, the tokenizer fingerprint,
optimizer/scheduler/AMP-scaler state, RNG state and the exact training position
(completed epochs plus micro-batches consumed in the current epoch).
"""

import hashlib
import os
import pickle
import random
from typing import Dict, Optional

import numpy as np
import torch


class _HmstCompatUnpickler(pickle.Unpickler):
    """Remaps old 'hmst' module/class names to 'mantis' during checkpoint loading."""
    _CLASS_RENAMES = {
        'HMSTConfig': 'MANTISConfig',
        'HMSTTokenizer': 'MANTISTokenizer',
        'HMSTInferenceEngine': 'MANTISInferenceEngine',
    }

    def find_class(self, module, name):
        if module.startswith('hmst'):
            module = 'mantis' + module[4:]
        name = self._CLASS_RENAMES.get(name, name)
        return super().find_class(module, name)


class _CompatPickle:
    """Pickle module stand-in that uses _HmstCompatUnpickler."""
    Unpickler = _HmstCompatUnpickler
    def __getattr__(self, name):
        return getattr(pickle, name)


def compat_load(path, *, map_location='cpu'):
    """Load a checkpoint, remapping old 'hmst' module paths to 'mantis'."""
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_CompatPickle())


def check_tokenizer(checkpoint: Dict, tokenizer, source: str) -> None:
    """Fail when a checkpoint was trained with a different vocabulary."""
    expected = checkpoint.get('tokenizer_fingerprint')
    if expected is not None and expected != tokenizer.fingerprint():
        raise ValueError(
            f"Tokenizer mismatch for {source}: checkpoint expects fingerprint {expected}, "
            f"got {tokenizer.fingerprint()}. Use the tokenizer saved with that checkpoint."
        )


def load_tokenizer(checkpoint_path: str, checkpoint: Dict, tokenizer_path: Optional[str] = None):
    """
    Load the tokenizer for a checkpoint: an explicit path, else the `tokenizer/`
    directory next to the checkpoint, else the built-in vocabulary. The result
    must match the fingerprint recorded in the checkpoint.
    """
    from mantis.tokenizer import MANTISTokenizer

    path = tokenizer_path or os.path.join(os.path.dirname(checkpoint_path), 'tokenizer')
    if os.path.isdir(path):
        tokenizer = MANTISTokenizer.load(path)
    elif tokenizer_path:
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    else:
        tokenizer = MANTISTokenizer()
    check_tokenizer(checkpoint, tokenizer, checkpoint_path)
    return tokenizer


def model_fingerprint(model, tokenizer) -> str:
    """
    Identity of the tokenizer and all backbone weights that produce memory
    embeddings. Hash one tensor at a time to avoid duplicating the model in RAM.
    """
    digest = hashlib.sha256(tokenizer.fingerprint().encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


def load_base_model(checkpoint_path: str, device: str = 'cpu', tokenizer_path: Optional[str] = None,
                    dtype: Optional[str] = None):
    """
    Load a Stage 1 base model in eval mode.

    Args:
        dtype: Serving dtype ('float32', 'float16', 'bfloat16'); None keeps the saved float32

    Returns:
        (model, tokenizer, checkpoint_dict)
    """
    from mantis.models.base_moe import BaseMoEModel

    checkpoint = compat_load(checkpoint_path)
    if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError(f"Not a base model checkpoint (needs 'config' and 'model_state_dict'): {checkpoint_path}")

    config = checkpoint['config']
    tokenizer = load_tokenizer(checkpoint_path, checkpoint, tokenizer_path)
    if config.base_moe.vocab_size != len(tokenizer):
        raise ValueError(
            f"Checkpoint vocab_size {config.base_moe.vocab_size} != tokenizer size {len(tokenizer)}"
        )

    model = BaseMoEModel.from_config(config.base_moe)
    model.load_state_dict(checkpoint['model_state_dict'])
    if dtype:
        if dtype != 'float32':
            # Preserve the checkpoint's identity before serving precision
            # rounds its weights differently from the Stage 2 index builder.
            model._source_fingerprint = model_fingerprint(model, tokenizer)
        model.to(DTYPES[dtype])
    model.to(device).eval()
    return model, tokenizer, checkpoint


def save_training_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    config,
    tokenizer,
    epoch: int,
    epoch_step: int,
    optimizer_step: int,
    global_step: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    **extra,
) -> None:
    """
    Save a resumable training checkpoint.

    Args:
        epoch: Number of fully completed epochs
        epoch_step: Micro-batches already consumed in epoch `epoch` (0 at an epoch boundary)
        optimizer: Pass None when its state is sharded (DeepSpeed ZeRO)
        extra: Additional fields (val_loss, best_val_loss, ...)
    """
    state = {
        'config': config,
        'tokenizer_fingerprint': tokenizer.fingerprint(),
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'epoch_step': epoch_step,
        'step': optimizer_step,
        'global_step': global_step,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    state.update(extra)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)


def restore_training_state(checkpoint: Dict, *, optimizer=None, scheduler=None, scaler=None) -> Dict:
    """
    Restore optimizer/scheduler/scaler/RNG state saved by save_training_checkpoint.

    Returns:
        Dict with epoch, epoch_step, step, global_step, best_val_loss,
        epochs_without_improvement and a list of warnings for missing state.
    """
    warnings = []
    for obj, key in ((optimizer, 'optimizer_state_dict'), (scheduler, 'scheduler_state_dict'),
                     (scaler, 'scaler_state_dict')):
        if obj is None:
            continue
        if key in checkpoint:
            obj.load_state_dict(checkpoint[key])
        else:
            warnings.append(f"No {key} in checkpoint; starting it fresh")

    rng = checkpoint.get('rng_state')
    if rng:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available() \
                and len(rng['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng['cuda'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'epoch_step': checkpoint.get('epoch_step', 0),
        'step': checkpoint.get('step', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'epochs_without_improvement': checkpoint.get('epochs_without_improvement', 0),
        'warnings': warnings,
    }
