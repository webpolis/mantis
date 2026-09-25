"""
Pre-tokenize datasets for efficient training.

Input files hold documents separated by blank lines. Uses HuggingFace
datasets for storage. Tokenize once, train many times.

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
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, Features, Sequence, Value
from mantis.data import encode_documents, iter_documents, pack_windows
from mantis.tokenizer import BPETokenizer, MANTISTokenizer, load_tokenizer


def _windows(input_file, tokenizer, seq_len, stride, source_mtime):
    """Yield packed windows; `source_mtime` only invalidates the datasets cache."""
    for window in pack_windows(encode_documents(iter_documents(input_file), tokenizer), seq_len, stride):
        yield {'input_ids': window[:-1], 'labels': window[1:]}


def preprocess_dataset(input_file, output_path, tokenizer, seq_len, stride):
    """
    Preprocess a text file into a tokenized dataset of packed windows.

    Documents are separated by blank lines; each gets one EOS. Windows hold
    seq_len inputs and seq_len shifted labels, cut every `stride` tokens,
    exactly as train.py builds them from raw text.
    """
    print(f"\n{'='*80}")
    print(f"Preprocessing: {input_file}")
    print(f"Output: {output_path}")
    print(f"Sequence length: {seq_len}, Stride: {stride}")
    print(f"{'='*80}\n")

    features = Features({
        'input_ids': Sequence(Value('int32')),
        'labels': Sequence(Value('int32')),
    })
    tokenized_dataset = Dataset.from_generator(
        _windows,
        features=features,
        gen_kwargs={
            'input_file': str(input_file),
            'tokenizer': tokenizer,
            'seq_len': seq_len,
            'stride': stride,
            'source_mtime': os.path.getmtime(input_file),
        },
    )
    if len(tokenized_dataset) == 0:
        raise ValueError(f"{input_file} has fewer than {seq_len + 1} tokens")

    print(f"\nSaving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)

    # Save preprocessing config so train.py can read seq_len/stride and verify the tokenizer
    with open(Path(output_path) / "preprocessing_config.json", 'w') as f:
        json.dump({
            "seq_len": seq_len,
            "stride": stride,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
        }, f)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Sequences: {len(tokenized_dataset):,}")
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
                       help='Existing tokenizer directory. Without it, one is created per --tokenizer-type')
    parser.add_argument('--tokenizer-type', choices=['bpe', 'mantis'], default='bpe',
                       help='Tokenizer to create: byte-level BPE trained on the input (default) or the '
                            'fixed trie tokenizer of the evolution format')
    parser.add_argument('--vocab-size', type=int, default=32768,
                       help='BPE vocabulary size (default: 32768)')
    parser.add_argument('--tokenizer-train-docs', type=int, default=100_000,
                       help='Documents the BPE tokenizer is trained on, 0 for all (default: 100000)')
    parser.add_argument('--tokenizer-save', type=str,
                       help='Where to save a created tokenizer (default: tokenizer/ next to --output)')

    # Sequence parameters
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length (default: 512)')
    parser.add_argument('--stride', type=int,
                       help='Stride for sequences (default: seq-len for non-overlapping)')

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
    if not 0 < stride <= args.seq_len:
        print(f"Error: --stride must be between 1 and --seq-len ({args.seq_len})")
        return

    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Loaded {type(tokenizer).__name__} from {args.tokenizer}: {len(tokenizer):,} tokens")
    else:
        if args.tokenizer_type == 'mantis':
            tokenizer = MANTISTokenizer()
        else:
            docs = f"{args.tokenizer_train_docs:,}" if args.tokenizer_train_docs else "all"
            print(f"Training a {args.vocab_size:,}-token BPE tokenizer on {docs} documents of {args.input}...")
            texts = itertools.islice(iter_documents(args.input), args.tokenizer_train_docs or None)
            tokenizer = BPETokenizer.train(texts, args.vocab_size)
        save_path = args.tokenizer_save or os.path.join(os.path.dirname(args.output.rstrip('/')), 'tokenizer')
        tokenizer.save(save_path)
        print(f"Tokenizer saved: {save_path} ({len(tokenizer):,} tokens)")

    # Preprocess training data
    preprocess_dataset(
        args.input,
        args.output,
        tokenizer,
        args.seq_len,
        stride,
    )

    # Preprocess validation data
    if args.val_input:
        preprocess_dataset(
            args.val_input,
            args.val_output,
            tokenizer,
            args.seq_len,
            args.seq_len,  # No overlap for validation
        )

    print(f"\n{'='*80}")
    print("All preprocessing complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
