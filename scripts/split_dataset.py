"""
Split Pre-tokenized Dataset into Train/Val

Validation is the tail of the (sequentially packed) dataset; windows that
would overlap it are dropped from train, so no token appears in both.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --input data/tokenized/train --test-size 0.05
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from mantis.data import split_packed


def split_dataset(
    input_path: str = 'data/tokenized/train',
    train_output: str = 'data/tokenized/train_split',
    val_output: str = 'data/tokenized/val',
    test_size: float = 0.1,
):
    """
    Split pre-tokenized dataset into train and validation sets.

    Args:
        input_path: Path to full tokenized dataset (from preprocess_data.py)
        train_output: Path to save training split
        val_output: Path to save validation split
        test_size: Fraction for validation (default 0.1 = 10%)
    """
    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            f"Have you tokenized your data? Run:\n"
            f"  python scripts/preprocess_data.py --input data/train.txt --output {input_path}"
        )

    config_path = input_dir / 'preprocessing_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} missing; re-run scripts/preprocess_data.py")
    prep = json.loads(config_path.read_text())

    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)
    print(f"Total sequences: {len(ds):,}")
    print(f"Splitting tail {test_size:.0%} for validation (seq_len={prep['seq_len']}, stride={prep['stride']})...")

    train_ds, val_ds = split_packed(ds, test_size, prep['seq_len'], prep['stride'])

    print(f"\nTrain sequences: {len(train_ds):,}")
    print(f"Val sequences: {len(val_ds):,}")
    print(f"Dropped at boundary: {len(ds) - len(train_ds) - len(val_ds):,}")

    for split_ds, output in ((train_ds, train_output), (val_ds, val_output)):
        print(f"Saving to: {output}")
        split_ds.save_to_disk(output)
        shutil.copy(config_path, Path(output) / 'preprocessing_config.json')

    print("\n✓ Dataset split complete!")
    print(f"\nUsage:")
    print(f"  python train.py {train_output} \\")
    print(f"      --pretokenized \\")
    print(f"      --val-file {val_output} \\")
    print(f"      --model-size micro")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a pre-tokenized dataset into train/val')
    parser.add_argument('--input', default='data/tokenized/train')
    parser.add_argument('--train-output', default='data/tokenized/train_split')
    parser.add_argument('--val-output', default='data/tokenized/val')
    parser.add_argument('--test-size', type=float, default=0.1)
    args = parser.parse_args()
    split_dataset(args.input, args.train_output, args.val_output, args.test_size)
