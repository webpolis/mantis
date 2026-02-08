"""
Pre-tokenize datasets for efficient training.

Uses HuggingFace datasets library with automatic caching.
Tokenize once, train many times - industry best practice.

Usage:
    # Tokenize training data
    python scripts/preprocess_data.py \\
        --input data/train.txt \\
        --output data/tokenized/train \\
        --tokenizer checkpoints/improved_train/tokenizer

    # Tokenize with validation data
    python scripts/preprocess_data.py \\
        --input data/train.txt \\
        --val-input data/val.txt \\
        --output data/tokenized/train \\
        --val-output data/tokenized/val \\
        --tokenizer checkpoints/improved_train/tokenizer
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from mantis.tokenizer import MANTISTokenizer
from tqdm import tqdm


def tokenize_and_chunk(examples, tokenizer, seq_len, stride):
    """
    Tokenize text and create fixed-length sequences.

    This function:
    1. Tokenizes all texts in the batch
    2. Concatenates all tokens
    3. Splits into fixed-length chunks with stride
    4. Creates input_ids and labels for next-token prediction
    """
    # Tokenize all texts
    all_tokens = []
    for text in examples['text']:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    # Create sequences with stride
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, stride):
        chunk = all_tokens[i:i + seq_len + 1]  # +1 for label
        if len(chunk) == seq_len + 1:
            sequences.append({
                'input_ids': chunk[:-1],
                'labels': chunk[1:]
            })

    # Transpose list of dicts to dict of lists
    if sequences:
        return {
            'input_ids': [s['input_ids'] for s in sequences],
            'labels': [s['labels'] for s in sequences]
        }
    else:
        return {'input_ids': [], 'labels': []}


def preprocess_dataset(input_file, output_path, tokenizer, seq_len, stride, num_proc=4):
    """
    Preprocess a text file into tokenized dataset.

    Args:
        input_file: Path to input text file
        output_path: Path to save tokenized dataset
        tokenizer: MANTISTokenizer instance
        seq_len: Sequence length
        stride: Stride for overlapping sequences
        num_proc: Number of processes for parallel tokenization
    """
    print(f"\n{'='*80}")
    print(f"Preprocessing: {input_file}")
    print(f"Output: {output_path}")
    print(f"Sequence length: {seq_len}, Stride: {stride}")
    print(f"{'='*80}\n")

    # Load raw text
    print("Loading text file...")
    dataset = load_dataset('text', data_files=str(input_file), split='train')
    print(f"Loaded {len(dataset):,} lines")

    # Tokenize and create sequences
    print("\nTokenizing and creating sequences...")
    print("(This will be cached - future runs will be instant!)")

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer, seq_len, stride),
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        num_proc=num_proc,
        desc="Tokenizing"
    )

    print(f"\nCreated {len(tokenized_dataset):,} training sequences")

    # Save to disk
    print(f"\nSaving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)

    # Show stats
    total_tokens = len(tokenized_dataset) * seq_len
    print(f"\n✓ Preprocessing complete!")
    print(f"  Sequences: {len(tokenized_dataset):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Saved to: {output_path}")

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Pre-tokenize datasets for efficient training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize single file (creates new tokenizer)
  python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train

  # Use existing tokenizer from checkpoint
  python scripts/preprocess_data.py \\
      --input data/train.txt \\
      --output data/tokenized/train \\
      --tokenizer checkpoints/improved_train/tokenizer

  # Process both train and validation
  python scripts/preprocess_data.py \\
      --input data/train.txt \\
      --val-input data/val.txt \\
      --output data/tokenized/train \\
      --val-output data/tokenized/val \\
      --tokenizer checkpoints/improved_train/tokenizer

  # Control sequence parameters
  python scripts/preprocess_data.py \\
      --input data/train.txt \\
      --output data/tokenized/train \\
      --seq-len 1024 \\
      --stride 512
        """
    )

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file (training data)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for tokenized dataset')
    parser.add_argument('--val-input', type=str,
                       help='Validation text file (optional)')
    parser.add_argument('--val-output', type=str,
                       help='Output directory for tokenized validation dataset')

    # Tokenizer
    parser.add_argument('--tokenizer', type=str,
                       help='Path to existing tokenizer. If not provided, creates new tokenizer')
    parser.add_argument('--tokenizer-save', type=str,
                       help='Path to save tokenizer (only used when creating new tokenizer)')

    # Sequence parameters
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length (default: 512)')
    parser.add_argument('--stride', type=int,
                       help='Stride for sequences (default: seq-len for non-overlapping)')

    # Performance
    parser.add_argument('--num-proc', type=int, default=4,
                       help='Number of processes for parallel tokenization (default: 4)')

    args = parser.parse_args()

    # Validate
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    if args.val_input and not Path(args.val_input).exists():
        print(f"Error: Validation file not found: {args.val_input}")
        return

    if args.val_input and not args.val_output:
        print("Error: --val-output required when --val-input is provided")
        return

    stride = args.stride or args.seq_len

    # Load or create tokenizer
    if args.tokenizer:
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = MANTISTokenizer.load(args.tokenizer)
        print(f"Loaded vocabulary: {len(tokenizer):,} tokens")
    else:
        print("Creating new tokenizer...")
        tokenizer = MANTISTokenizer()
        print(f"Vocabulary size: {len(tokenizer):,} tokens")

        # Save tokenizer if requested
        if args.tokenizer_save:
            print(f"Saving tokenizer to {args.tokenizer_save}...")
            tokenizer.save(args.tokenizer_save)

    # Preprocess training data
    preprocess_dataset(
        args.input,
        args.output,
        tokenizer,
        args.seq_len,
        stride,
        args.num_proc
    )

    # Preprocess validation data
    if args.val_input:
        preprocess_dataset(
            args.val_input,
            args.val_output,
            tokenizer,
            args.seq_len,
            args.seq_len,  # No overlap for validation
            args.num_proc
        )

    print(f"\n{'='*80}")
    print("All preprocessing complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
