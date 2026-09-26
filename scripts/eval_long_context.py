#!/usr/bin/env python3
"""
Needle-in-a-haystack evaluation for a Stage 1 checkpoint.

A pass key sentence is hidden at several depths of a haystack of filler
text, the model is asked for the key, and the answer is checked. Accuracy
per depth and per context length, prefill time and peak GPU memory are
reported; the tokenized prompt is never truncated (a prompt longer than the
model window is an error, not a silent cut).

    uv run scripts/eval_long_context.py checkpoints/ext/best_model.pt \\
        --lengths 4096 32768 262144 --depths 0 0.25 0.5 0.75 1.0 --samples 5

The prompt format is plain text (no chat template): the haystack, the
needle sentence, a question, and the answer prefix "The pass key is".
Held-out seeds keep the keys disjoint from any synthetic training data.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mantis.inference.generation import generate_tokens
from mantis.utils.checkpoints import load_base_model

FILLER = [
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
    "Rivers run to the sea and the wind moves the clouds across the hills before the evening comes.",
    "A quiet town keeps its streets clean and its lamps lit through the long nights of winter.",
    "The market opens at dawn with bread, fruit and fish, and closes when the last stall is empty.",
]


def build_prompt(tokenizer, length, depth, key, rng):
    needle = f" The pass key is {key}. Remember it. "
    question = "\n\nWhat is the pass key? The pass key is"
    budget = length - len(tokenizer.encode(needle)) - len(tokenizer.encode(question)) - 8
    filler = []
    while len(tokenizer.encode(" ".join(filler))) < budget:
        filler.append(rng.choice(FILLER))
    while len(tokenizer.encode(" ".join(filler))) > budget:
        filler.pop()
    cut = int(len(filler) * depth)
    text = " ".join(filler[:cut]) + needle + " ".join(filler[cut:]) + question
    return tokenizer.encode(text)


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack over context lengths and depths")
    parser.add_argument('checkpoint')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--lengths', type=int, nargs='+', default=[4096, 32768])
    parser.add_argument('--depths', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--samples', type=int, default=3, help='Keys per (length, depth)')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--rope-factor', type=float,
                        help='Override the checkpoint\'s YaRN factor for an inference-only extrapolation')
    parser.add_argument('--output', help='Write the per-example records as JSON')
    args = parser.parse_args()

    model, tokenizer, checkpoint = load_base_model(args.checkpoint, args.device, args.tokenizer_path, dtype=args.dtype)
    if args.rope_factor is not None:
        config = checkpoint['config'].base_moe
        config.rope_factor = args.rope_factor
        config.max_seq_len = max(config.max_seq_len, max(args.lengths))
        extended = type(model).from_config(config)
        extended.load_state_dict(model.state_dict())
        model = extended.to(args.device).to(next(model.parameters()).dtype).eval()
    if max(args.lengths) > model.max_seq_len:
        sys.exit(f"Longest length {max(args.lengths)} exceeds the model window {model.max_seq_len}; "
                 "extend it with --rope-factor")
    print(f"Model window {model.max_seq_len}, layout {model.attention_layout()}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    rng = random.Random(args.seed)
    records = []
    for length in args.lengths:
        for depth in args.depths:
            correct = 0
            for _ in range(args.samples):
                key = str(rng.randint(100000, 999999))
                prompt = build_prompt(tokenizer, length, depth, key, rng)
                if len(prompt) > model.max_seq_len:
                    sys.exit(f"Prompt of {len(prompt)} tokens exceeds the window {model.max_seq_len}")
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats()
                start = time.time()
                tokens = [t for t, _ in generate_tokens(model, prompt, max_new_tokens=8, temperature=0.0,
                                                       banned_ids=tokenizer.non_generable_ids,
                                                       stop_ids=[tokenizer.eos_token_id])]
                elapsed = time.time() - start
                answer = tokenizer.decode(tokens)
                hit = key in answer.replace(" ", "")
                correct += hit
                records.append({'length': length, 'prompt_tokens': len(prompt), 'depth': depth, 'key': key,
                                'answer': answer, 'correct': hit, 'seconds': elapsed,
                                'peak_gb': torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith('cuda') else None})
            last = records[-1]
            print(f"length {length:>8,} depth {depth:4.2f}: {correct}/{args.samples} correct, "
                  f"{last['seconds']:6.1f} s, peak {last['peak_gb'] or 0:5.2f} GB, e.g. {last['answer']!r}")
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(records, f, indent=1)
    total = sum(r['correct'] for r in records)
    print(f"\nOverall: {total}/{len(records)} correct")


if __name__ == '__main__':
    main()
