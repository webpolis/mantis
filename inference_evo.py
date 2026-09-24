"""
Evolution Inference Engine for MANTIS

Tick-by-tick generation of evolution simulation traces.
Designed as an importable module for web apps with CLI as secondary interface.

Usage (Python):
    from inference_evo import EvoInferenceEngine

    engine = EvoInferenceEngine("checkpoints/evo_train/best_model.pt")

    # Generate a new world
    for tick in engine.generate_world(seed=42, temperature=0.7):
        print(tick)

    # Continue from partial state
    for tick in engine.continue_trace(existing_trace, max_ticks=10):
        send_to_client(tick)

Usage (CLI):
    python inference_evo.py checkpoints/evo_train/best_model.pt --new-world --seed 42 --max-ticks 100
    python inference_evo.py checkpoints/evo_train/best_model.pt --continue trace.txt --max-ticks 50
    python inference_evo.py checkpoints/evo_train/best_model.pt --prompt "=EPOCH 1 1000 W0"
"""

import os
import subprocess

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        gpu_names = result.stdout.strip().split('\n')
        problematic_gpus = ['RTX 30', 'RTX 40', 'A4000', 'A5000', 'A6000']
        detected_buggy = [name for name in gpu_names if any(gpu in name for gpu in problematic_gpus)]
        if detected_buggy:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
            os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
except Exception:
    pass

import torch
import argparse
import sys
import time
from typing import Optional, Generator

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from mantis.inference.generation import generate_tokens
from mantis.utils.checkpoints import load_base_model


class EvoInferenceEngine:
    """Tick-by-tick evolution simulation inference engine."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantize = quantize
        if quantize == 'int8' and self.device != 'cpu':
            print("INT8 dynamic quantization only has CPU kernels; running on CPU")
            self.device = 'cpu'

        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")

        self.model, self.tokenizer, checkpoint = load_base_model(checkpoint_path, self.device)
        self.max_seq_len = self.model.max_seq_len

        if self.quantize == 'int8':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            print("✓ Model quantized to INT8")
        elif self.quantize == 'float16' and self.device == 'cuda':
            self.model = self.model.half()
            print("✓ Model converted to FP16")

        self.separator_id = self.tokenizer.vocab['---']
        self.eos_id = self.tokenizer.eos_token_id
        self.banned_ids = self.tokenizer.non_generable_ids

        param_counts = self.model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M parameters")
        print(f"Vocabulary: {len(self.tokenizer):,} tokens")
        print(f"Context window: {self.max_seq_len} tokens")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    def _stream_ticks(
        self,
        context: str,
        max_ticks: int,
        max_tokens_per_tick: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """
        Decode one continuous stream and cut it into ticks at each `---`.

        A tick also ends after `max_tokens_per_tick` tokens. EOS ends the world.
        The KV cache carries across ticks; the shared decoder keeps the context
        inside the model's window.
        """
        prompt_ids = self.tokenizer.encode(context)
        stream = generate_tokens(
            self.model, prompt_ids, max_ticks * max_tokens_per_tick,
            temperature=temperature, top_k=top_k, top_p=top_p,
            banned_ids=self.banned_ids, stop_ids=[self.eos_id],
        )

        tick = []
        emitted = 0
        for token, _ in stream:
            if token == self.eos_id:
                break
            tick.append(token)
            if token == self.separator_id or len(tick) >= max_tokens_per_tick:
                yield self.tokenizer.decode(tick)
                emitted += 1
                tick = []
                if emitted >= max_ticks:
                    return
        if tick:
            yield self.tokenizer.decode(tick)

    def generate_tick(
        self,
        context: str,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate a single tick from context, up to and including its `---` separator."""
        return next(self._stream_ticks(context, 1, max_tokens, temperature, top_p, top_k), "")

    def generate_world(
        self,
        seed: int = 0,
        max_ticks: int = 100,
        max_tokens_per_tick: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Generator[str, None, None]:
        """Generate a new world from scratch, yielding tick-by-tick.

        Starts with `=EPOCH 1 1000 W{seed}` prompt (v2 compact format).
        """
        yield from self._stream_ticks(
            f"=EPOCH 1 1000 W{seed}\n", max_ticks, max_tokens_per_tick, temperature, top_p, top_k,
        )

    def continue_trace(
        self,
        partial_trace: str,
        max_ticks: int = 10,
        max_tokens_per_tick: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Generator[str, None, None]:
        """Continue from an existing simulation trace, yielding new ticks."""
        yield from self._stream_ticks(
            partial_trace, max_ticks, max_tokens_per_tick, temperature, top_p, top_k,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MANTIS Evolution Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    # Mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--new-world', action='store_true',
                            help='Generate a new world from scratch')
    mode_group.add_argument('--continue', dest='continue_file', type=str,
                            help='Continue from a partial trace file')
    mode_group.add_argument('--prompt', type=str,
                            help='Generate from a custom prompt string')

    # Generation parameters
    parser.add_argument('--max-ticks', type=int, default=100,
                        help='Maximum ticks to generate (default: 100)')
    parser.add_argument('--max-tokens-per-tick', type=int, default=4096,
                        help='Maximum tokens per tick (default: 4096)')
    parser.add_argument('--seed', type=int, default=0,
                        help='World seed for --new-world mode (default: 0)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling threshold (default: 0.9)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling threshold (default: 50)')

    # Output
    parser.add_argument('--output', type=str,
                        help='Write output to file instead of stdout')

    # System
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device (default: auto-detect)')
    parser.add_argument('--quantize', type=str, choices=['int8', 'float16'],
                        help='Quantization mode')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    engine = EvoInferenceEngine(
        args.checkpoint, device=args.device, quantize=args.quantize,
    )

    gen_kwargs = dict(
        max_tokens_per_tick=args.max_tokens_per_tick,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    out_file = open(args.output, 'w') if args.output else None

    def emit(text):
        if out_file:
            out_file.write(text)
            out_file.flush()
        else:
            print(text, end='', flush=True)

    try:
        t0 = time.time()
        tick_count = 0

        if args.new_world:
            print(f"\nGenerating new world (seed={args.seed})...\n", file=sys.stderr)
            for tick in engine.generate_world(
                seed=args.seed, max_ticks=args.max_ticks, **gen_kwargs,
            ):
                emit(tick)
                tick_count += 1

        elif args.continue_file:
            if not os.path.exists(args.continue_file):
                print(f"Error: Trace file not found: {args.continue_file}")
                sys.exit(1)
            with open(args.continue_file, 'r') as f:
                partial_trace = f.read()
            print(f"\nContinuing trace ({len(partial_trace)} chars)...\n", file=sys.stderr)
            for tick in engine.continue_trace(
                partial_trace, max_ticks=args.max_ticks, **gen_kwargs,
            ):
                emit(tick)
                tick_count += 1

        elif args.prompt:
            print(f"\nGenerating from prompt...\n", file=sys.stderr)
            for tick in engine.continue_trace(
                args.prompt, max_ticks=args.max_ticks, **gen_kwargs,
            ):
                emit(tick)
                tick_count += 1

        elapsed = time.time() - t0
        print(f"\n\nGenerated {tick_count} ticks in {elapsed:.1f}s", file=sys.stderr)

    finally:
        if out_file:
            out_file.close()


if __name__ == '__main__':
    main()
