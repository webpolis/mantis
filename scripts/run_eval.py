#!/usr/bin/env python3
"""
CLI Script to run MANTIS model evaluations.

Usage:
    python scripts/run_eval.py checkpoints/model.pt --benchmarks mmlu truthfulqa
    python scripts/run_eval.py checkpoints/model.pt --all --output results.json
"""

import argparse
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mantis import BaseMoEModel
from mantis.tokenizer import MANTISTokenizer
from mantis.utils.checkpoints import compat_load
from evaluation import EvaluationHarness


def load_model(checkpoint_path: str, tokenizer_path: str, device: str):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = compat_load(checkpoint_path)
    config = checkpoint['config']

    # Load tokenizer
    tokenizer = MANTISTokenizer.load(tokenizer_path)
    config.base_moe.vocab_size = len(tokenizer)

    # Load model
    model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout,
        load_balance_weight=config.base_moe.load_balance_weight
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model, tokenizer


def load_demo_datasets():
    """Load demo datasets for testing."""
    datasets = {
        'mmlu': [
            {
                "question": "What is the capital of France?",
                "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
                "answer": "B",
                "subject": "geography"
            },
            {
                "question": "What is 2 + 2?",
                "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
                "answer": "B",
                "subject": "math"
            }
        ],
        'truthfulqa': [
            {
                "question": "What happens if you break a mirror?",
                "best_answer": "Nothing special happens if you break a mirror.",
                "incorrect_answers": [
                    "You get 7 years of bad luck",
                    "You will have bad luck"
                ]
            }
        ],
        'gsm8k': [
            {
                "question": "If John has 3 apples and buys 2 more, how many does he have?",
                "answer": "5"
            }
        ]
    }

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description='Run MANTIS model evaluation on standard benchmarks'
    )

    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer directory')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       choices=['mmlu', 'truthfulqa', 'humaneval', 'gsm8k'],
                       help='Specific benchmarks to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--output', type=str,
                       help='Path to save results JSON')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo datasets (for testing)')

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.benchmarks:
        print("Error: Must specify either --all or --benchmarks")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        return 1

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    # Initialize evaluation harness
    harness = EvaluationHarness(model, tokenizer, device)

    # Load datasets
    if args.demo:
        print("\n⚠️  Using demo datasets (small test samples)")
        datasets = load_demo_datasets()
    else:
        print("\n⚠️  Note: Real benchmark datasets not loaded.")
        print("   To use real benchmarks, download datasets and modify this script.")
        print("   Using demo datasets for now.\n")
        datasets = load_demo_datasets()

    # Filter benchmarks
    if not args.all:
        datasets = {k: v for k, v in datasets.items() if k in args.benchmarks}

    # Run evaluations
    print(f"\n{'='*80}")
    print("MANTIS MODEL EVALUATION")
    print(f"{'='*80}\n")
    print(f"Model: {args.checkpoint}")
    print(f"Benchmarks: {list(datasets.keys())}")
    print(f"Device: {device}\n")

    results = harness.run_all_benchmarks(datasets)

    # Generate report
    harness.generate_report(results, output_path=args.output)

    print("\n✓ Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
