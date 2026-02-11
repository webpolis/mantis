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
import torch.nn.functional as F
import argparse
import sys
import time
from typing import Optional, Generator

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from mantis.models import BaseMoEModel
from mantis.tokenizer import MANTISTokenizer
from mantis.utils.checkpoints import compat_load


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

        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")

        checkpoint = compat_load(checkpoint_path, map_location=self.device)

        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing 'config' field")
        config = checkpoint['config']

        self.model = BaseMoEModel(
            vocab_size=config.base_moe.vocab_size,
            d_model=config.base_moe.d_model,
            n_layers=config.base_moe.n_layers,
            n_heads=config.base_moe.n_heads,
            d_ff=config.base_moe.d_ff,
            n_experts=config.base_moe.n_experts,
            top_k=config.base_moe.top_k,
            max_seq_len=config.base_moe.max_seq_len,
            dropout=config.base_moe.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.max_seq_len = config.base_moe.max_seq_len

        if self.quantize == 'int8':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            print("✓ Model quantized to INT8")
        elif self.quantize == 'float16' and self.device == 'cuda':
            self.model = self.model.half()
            print("✓ Model converted to FP16")

        # Load tokenizer
        checkpoint_dir = os.path.dirname(checkpoint_path)
        tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')
        if os.path.exists(tokenizer_path):
            self.tokenizer = MANTISTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = MANTISTokenizer()

        # Cache the separator token id
        self.separator_id = self.tokenizer.vocab.get('---')
        self.eos_id = self.tokenizer.eos_token_id

        param_counts = self.model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M parameters")
        print(f"Vocabulary: {len(self.tokenizer):,} tokens")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    def _apply_sampling_filters(self, logits, top_k=50, top_p=0.9):
        """Apply top-k and top-p filtering to logits."""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[-1]
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float('-inf')),
                logits,
            )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.numel() > 1:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        return logits

    @torch.no_grad()
    def generate_tick(
        self,
        context: str,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate a single tick from context.

        Generates tokens until `---` separator or EOS is produced.
        Returns the generated tick text including the `---` separator.
        """
        self.model.eval()

        input_ids = self.tokenizer.encode(context, add_special_tokens=False)

        # Truncate context from the left if too long
        max_context = self.max_seq_len - max_tokens
        if len(input_ids) > max_context:
            input_ids = input_ids[-max_context:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        is_greedy = (temperature == 0.0)
        if is_greedy:
            temperature = 1.0

        generated_tokens = []
        past_key_values = None

        for _ in range(max_tokens):
            if past_key_values is None:
                model_input = input_tensor[:, -self.max_seq_len:]
            else:
                model_input = input_tensor[:, -1:]

            output = self.model(model_input, past_key_values=past_key_values, use_cache=True)
            logits = output['logits']
            past_key_values = output['past_key_values']

            # Truncate KV cache if needed
            if past_key_values[0][0].size(1) >= self.max_seq_len:
                past_key_values = [
                    (k[:, -self.max_seq_len + 1:], v[:, -self.max_seq_len + 1:])
                    for k, v in past_key_values
                ]

            next_token_logits = logits[0, -1, :] / temperature

            if is_greedy:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = self._apply_sampling_filters(
                    next_token_logits, top_k=top_k, top_p=top_p,
                )
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            token_id = next_token.item()
            generated_tokens.append(token_id)

            if token_id == self.eos_id:
                break

            # Stop on --- separator
            if token_id == self.separator_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        Yields each generated tick as a string.
        """
        context = f"=EPOCH 1 1000 W{seed}\n"

        for _ in range(max_ticks):
            tick = self.generate_tick(
                context,
                max_tokens=max_tokens_per_tick,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if not tick.strip():
                break

            yield tick

            # Append to context, left-truncate if needed
            context += tick
            # Keep context within model's capacity (leave room for generation)
            max_context_chars = self.max_seq_len * 4  # rough char estimate
            if len(context) > max_context_chars:
                # Find a tick separator to cut at
                cut_point = context.find('---', len(context) - max_context_chars)
                if cut_point > 0:
                    context = context[cut_point:]

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
        context = partial_trace

        for _ in range(max_ticks):
            tick = self.generate_tick(
                context,
                max_tokens=max_tokens_per_tick,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if not tick.strip():
                break

            yield tick

            context += tick
            max_context_chars = self.max_seq_len * 4
            if len(context) > max_context_chars:
                cut_point = context.find('---', len(context) - max_context_chars)
                if cut_point > 0:
                    context = context[cut_point:]


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
