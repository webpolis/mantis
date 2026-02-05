"""
Download and Prepare Training Dataset

Downloads WikiText-103 dataset and prepares it for training.
WikiText-103 contains ~100M tokens from Wikipedia articles.
"""

from datasets import load_dataset
import os
from tqdm import tqdm


def download_wikitext(output_dir='./data', split='train'):
    """
    Download WikiText-103 dataset.

    Args:
        output_dir: Directory to save processed data
        split: Dataset split ('train', 'validation', 'test')

    Returns:
        Path to saved text file
    """
    print(f"\nDownloading WikiText-103 ({split} split)...")
    print("=" * 80)

    # Download dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Merge all text
    output_file = os.path.join(output_dir, f'wikitext-103-{split}.txt')

    print(f"\nProcessing {len(dataset)} documents...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Writing"):
            text = example['text'].strip()
            if text:  # Skip empty lines
                f.write(text + '\n')

    # Get file stats
    file_size = os.path.getsize(output_file)
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        num_chars = len(content)
        num_words = len(content.split())

    print(f"\n{'=' * 80}")
    print(f"Dataset saved: {output_file}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Characters: {num_chars:,}")
    print(f"Words: {num_words:,}")
    print(f"{'=' * 80}\n")

    return output_file


def download_tiny_stories(output_dir='./data'):
    """
    Download TinyStories dataset (smaller, simpler alternative).

    TinyStories contains ~2M short stories generated for children.
    Good for faster training while still having substantial data.

    Args:
        output_dir: Directory to save processed data

    Returns:
        Path to saved text file
    """
    print(f"\nDownloading TinyStories dataset...")
    print("=" * 80)

    # Download dataset (train split)
    dataset = load_dataset('roneneldan/TinyStories', split='train')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'tinystories-train.txt')

    print(f"\nProcessing {len(dataset)} stories...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Writing"):
            text = example['text'].strip()
            if text:
                f.write(text + '\n\n')  # Double newline between stories

    # Get file stats
    file_size = os.path.getsize(output_file)
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        num_chars = len(content)
        num_words = len(content.split())

    print(f"\n{'=' * 80}")
    print(f"Dataset saved: {output_file}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Characters: {num_chars:,}")
    print(f"Words: {num_words:,}")
    print(f"{'=' * 80}\n")

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download training dataset')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'tinystories'],
                        help='Dataset to download')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split (for wikitext)')

    args = parser.parse_args()

    if args.dataset == 'wikitext':
        # Download all splits
        train_file = download_wikitext(args.output_dir, 'train')
        val_file = download_wikitext(args.output_dir, 'validation')

        print("\nDownload complete!")
        print(f"Training data: {train_file}")
        print(f"Validation data: {val_file}")

    elif args.dataset == 'tinystories':
        train_file = download_tiny_stories(args.output_dir)

        print("\nDownload complete!")
        print(f"Training data: {train_file}")
