"""
Split Pre-tokenized Dataset into Train/Val

Usage:
    python scripts/split_dataset.py
"""

from datasets import load_from_disk
from pathlib import Path

def split_dataset(
    input_path: str = 'data/tokenized/train',
    train_output: str = 'data/tokenized/train_split',
    val_output: str = 'data/tokenized/val',
    test_size: float = 0.1,
    seed: int = 42
):
    """
    Split pre-tokenized dataset into train and validation sets.

    Args:
        input_path: Path to full tokenized dataset
        train_output: Path to save training split
        val_output: Path to save validation split
        test_size: Fraction for validation (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    # Validate input path
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            f"Have you tokenized your data? Run:\n"
            f"  python scripts/preprocess_data.py --input data/train.txt --output {input_path}"
        )
    if not input_dir.is_dir():
        raise ValueError(f"Expected directory, got file: {input_path}")

    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)

    print(f"Total sequences: {len(ds):,}")
    print(f"Splitting with test_size={test_size} (seed={seed})...")

    split = ds.train_test_split(test_size=test_size, seed=seed)

    train_ds = split['train']
    val_ds = split['test']

    print(f"\nTrain sequences: {len(train_ds):,}")
    print(f"Val sequences: {len(val_ds):,}")

    # Save splits
    print(f"\nSaving train split to: {train_output}")
    train_ds.save_to_disk(train_output)

    print(f"Saving validation split to: {val_output}")
    val_ds.save_to_disk(val_output)

    print("\n✓ Dataset split complete!")
    print(f"\nUsage:")
    print(f"  python train.py {train_output} \\")
    print(f"      --pretokenized \\")
    print(f"      --val-file {val_output} \\")
    print(f"      --model-size micro")


if __name__ == '__main__':
    split_dataset()
