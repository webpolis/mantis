#!/usr/bin/env python3
"""
CLI Script to run MANTIS model evaluations.

Usage:
    python scripts/run_eval.py checkpoints/model.pt --benchmarks mmlu truthfulqa --limit 200
    python scripts/run_eval.py checkpoints/model.pt --all --output results.json
    python scripts/run_eval.py checkpoints/model.pt --all --demo

    # Full MANTIS engine (routing, memory, critic) from a Stage 3 policy
    python scripts/run_eval.py checkpoints/model.pt --all --policy-checkpoint ckpt/meta_controller_rl.pt
"""

import argparse
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mantis.inference.engine import MANTISInferenceEngine
from mantis.utils.checkpoints import load_base_model
from evaluation import EvaluationHarness

BENCHMARKS = ['mmlu', 'truthfulqa', 'humaneval', 'gsm8k']


def load_demo_datasets():
    """Tiny built-in samples for smoke tests."""
    return {
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
                "correct_answers": ["If you break a mirror, nothing in particular happens."],
                "incorrect_answers": [
                    "You get 7 years of bad luck",
                    "You will have bad luck"
                ]
            }
        ],
        'humaneval': [
            {
                "task_id": "Demo/0",
                "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
                "entry_point": "add"
            }
        ],
        'gsm8k': [
            {
                "question": "If John has 3 apples and buys 2 more, how many does he have?",
                "answer": "5"
            }
        ]
    }


def load_hub_dataset(name: str, limit: int = None):
    """Load a benchmark from the HuggingFace Hub in runner format."""
    from datasets import load_dataset

    split = lambda base: f"{base}[:{limit}]" if limit else base
    if name == 'mmlu':
        ds = load_dataset('cais/mmlu', 'all', split=split('test'))
        return [{
            'question': ex['question'],
            'choices': [f"{letter}) {choice}" for letter, choice in zip('ABCD', ex['choices'])],
            'answer': 'ABCD'[ex['answer']],
            'subject': ex['subject'],
        } for ex in ds]
    if name == 'truthfulqa':
        ds = load_dataset('truthful_qa', 'generation', split=split('validation'))
        return [{
            'question': ex['question'],
            'best_answer': ex['best_answer'],
            'correct_answers': ex['correct_answers'],
            'incorrect_answers': ex['incorrect_answers'],
        } for ex in ds]
    if name == 'humaneval':
        ds = load_dataset('openai_humaneval', split=split('test'))
        return [{k: ex[k] for k in ('task_id', 'prompt', 'test', 'entry_point')} for ex in ds]
    if name == 'gsm8k':
        ds = load_dataset('gsm8k', 'main', split=split('test'))
        return [{'question': ex['question'], 'answer': ex['answer']} for ex in ds]
    raise ValueError(f"Unknown benchmark: {name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MANTIS model evaluation on standard benchmarks'
    )

    parser.add_argument('checkpoint', type=str,
                        help='Path to the Stage 1 model checkpoint')
    parser.add_argument('--tokenizer', type=str,
                        help='Tokenizer directory (default: tokenizer/ next to the checkpoint)')
    parser.add_argument('--benchmarks', type=str, nargs='+', choices=BENCHMARKS,
                        help='Specific benchmarks to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmarks')
    parser.add_argument('--limit', type=int,
                        help='Evaluate only the first N examples of each benchmark')
    parser.add_argument('--output', type=str,
                        help='Path to save results JSON')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--demo', action='store_true',
                        help='Use built-in demo samples instead of downloading benchmarks')

    # Full engine
    parser.add_argument('--policy-checkpoint', type=str,
                        help='Stage 3 policy: evaluate the full MANTIS engine instead of the bare model')
    parser.add_argument('--memory-checkpoint', type=str, help='Stage 2 memory system (full engine)')
    parser.add_argument('--semantic-store', type=str, help='Stage 2 semantic store prefix (full engine)')
    parser.add_argument('--critic-checkpoint', type=str, help='Stage 4 critic (full engine)')

    args = parser.parse_args()

    if not args.all and not args.benchmarks:
        print("Error: Must specify either --all or --benchmarks")
        return 1
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.policy_checkpoint or args.memory_checkpoint or args.semantic_store or args.critic_checkpoint:
        model = MANTISInferenceEngine.from_checkpoints(
            base_checkpoint=args.checkpoint,
            policy_checkpoint=args.policy_checkpoint,
            memory_checkpoint=args.memory_checkpoint,
            semantic_store=args.semantic_store,
            critic_checkpoint=args.critic_checkpoint,
            tokenizer_path=args.tokenizer,
            device=device,
        )
        tokenizer = model.tokenizer
    else:
        model, tokenizer, _ = load_base_model(args.checkpoint, device, args.tokenizer)
    print("✓ Model loaded successfully")

    harness = EvaluationHarness(model, tokenizer, device)

    names = BENCHMARKS if args.all else args.benchmarks
    if args.demo:
        print("\n⚠️  Using demo datasets (small test samples)")
        demo = load_demo_datasets()
        datasets = {n: demo[n][:args.limit] if args.limit else demo[n] for n in names}
    else:
        try:
            datasets = {n: load_hub_dataset(n, args.limit) for n in names}
        except ImportError:
            print("Error: real benchmarks need the 'datasets' library (pip install datasets), or pass --demo")
            return 1

    print(f"\n{'='*80}")
    print("MANTIS MODEL EVALUATION")
    print(f"{'='*80}\n")
    print(f"Model: {args.checkpoint}")
    print(f"Benchmarks: {list(datasets.keys())}")
    print(f"Device: {device}\n")

    results = harness.run_all_benchmarks(datasets)
    harness.generate_report(results, output_path=args.output)

    print("\n✓ Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
