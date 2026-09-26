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

import torch
import argparse
import sys
import time
import uuid
from typing import Optional, Generator

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from mantis.inference.generation import generate_tokens
from mantis.inference.engine import MANTISInferenceEngine
from mantis.utils.checkpoints import load_base_model


class EvoInferenceEngine:
    """Tick-by-tick evolution simulation inference engine."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        quantize: Optional[str] = None,
        policy_checkpoint: Optional[str] = None,
        memory_checkpoint: Optional[str] = None,
        semantic_store: Optional[str] = None,
        critic_checkpoint: Optional[str] = None,
        memory_dir: Optional[str] = None,
        route_policy: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantize = quantize
        if quantize == 'int8' and self.device != 'cpu':
            print("INT8 dynamic quantization only has CPU kernels; running on CPU")
            self.device = 'cpu'

        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")

        self.full_engine = None
        self.last_result = None
        if policy_checkpoint:
            if quantize == 'int8':
                raise ValueError("INT8 quantization is unavailable with full evolution inference")
            self.full_engine = MANTISInferenceEngine.from_checkpoints(
                base_checkpoint=checkpoint_path, policy_checkpoint=policy_checkpoint,
                memory_checkpoint=memory_checkpoint, semantic_store=semantic_store,
                critic_checkpoint=critic_checkpoint, memory_dir=memory_dir,
                device=self.device, dtype='float16' if quantize == 'float16' and self.device == 'cuda' else None,
            )
            missing = [name for name, component in (
                ('episodic memory', self.full_engine.episodic),
                ('semantic memory', self.full_engine.semantic),
                ('critic', self.full_engine.critic),
            ) if component is None]
            if missing or self.full_engine.config.prompt_format != 'raw':
                self.full_engine.close()
                if missing:
                    raise ValueError(f"Full evolution inference requires {', '.join(missing)} checkpoints")
                raise ValueError("Evolution inference requires a raw-format generator checkpoint")
            if route_policy is not None:
                if route_policy not in ('learned', 'always', 'never', 'bypass'):
                    self.full_engine.close()
                    raise ValueError(f"Unknown route policy: {route_policy}")
                self.full_engine.config.route_policy = route_policy
            self.full_engine.config.prompt_format = 'trace'
            self.model, self.tokenizer = self.full_engine.base, self.full_engine.tokenizer
            checkpoint = {}
        else:
            if any((memory_checkpoint, semantic_store, critic_checkpoint, memory_dir, route_policy)):
                raise ValueError("A policy checkpoint is required for full evolution inference")
            self.model, self.tokenizer, checkpoint = load_base_model(checkpoint_path, self.device)
        self.max_seq_len = self.model.max_seq_len

        if self.full_engine is None and self.quantize == 'int8':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            print("✓ Model quantized to INT8")
        elif self.full_engine is None and self.quantize == 'float16' and self.device == 'cuda':
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

    def close(self) -> None:
        if self.full_engine is not None:
            self.full_engine.close()

    def _full_ticks(
        self, context: str, max_ticks: int, max_tokens_per_tick: int,
        temperature: float, top_p: float, top_k: int, namespace: Optional[str],
    ) -> Generator[str, None, None]:
        """Route, retrieve and verify each tick through the full engine."""
        namespace = namespace or f"evo-{uuid.uuid4().hex}"
        context_ids = self.tokenizer.encode(context)
        context_limit = max(1, self.max_seq_len // 3)
        if len(context_ids) > context_limit:
            self.full_engine.ingest(context, namespace=namespace, source='document')
        context_ids = context_ids[-context_limit:]

        for _ in range(max_ticks):
            result = self.full_engine.generate(
                self.tokenizer.decode(context_ids), max_length=max_tokens_per_tick,
                temperature=temperature, top_p=top_p, top_k=top_k,
                stop_ids=[self.separator_id], namespace=namespace,
            )
            self.last_result = result
            if result['abstained']:
                return
            tick = result['response']
            if result['stop_token_id'] == self.separator_id:
                tick += self.tokenizer.decode([self.separator_id])
            if tick:
                yield tick
            if result['stop_token_id'] != self.separator_id:
                return
            context_ids = (context_ids + self.tokenizer.encode(tick))[-context_limit:]

    def _stream_ticks(
        self,
        context: str,
        max_ticks: int,
        max_tokens_per_tick: int,
        temperature: float,
        top_p: float,
        top_k: int,
        namespace: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate ticks with full inference when configured, otherwise stream.

        The backbone path cuts a continuous stream at `---`; it also ends a
        tick after `max_tokens_per_tick` tokens. EOS ends the world.
        """
        if self.full_engine is not None:
            yield from self._full_ticks(
                context, max_ticks, max_tokens_per_tick, temperature, top_p, top_k, namespace,
            )
            return

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
        namespace: Optional[str] = None,
    ) -> str:
        """Generate a single tick from context, up to and including its `---` separator."""
        return next(self._stream_ticks(context, 1, max_tokens, temperature, top_p, top_k, namespace), "")

    def generate_world(
        self,
        seed: int = 0,
        max_ticks: int = 100,
        max_tokens_per_tick: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        namespace: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate a new world from scratch, yielding tick-by-tick.

        Starts with `=EPOCH 1 1000 W{seed}` prompt (v2 compact format).
        """
        yield from self._stream_ticks(
            f"=EPOCH 1 1000 W{seed}\n", max_ticks, max_tokens_per_tick, temperature, top_p, top_k, namespace,
        )

    def continue_trace(
        self,
        partial_trace: str,
        max_ticks: int = 10,
        max_tokens_per_tick: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        namespace: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Continue from an existing simulation trace, yielding new ticks."""
        yield from self._stream_ticks(
            partial_trace, max_ticks, max_tokens_per_tick, temperature, top_p, top_k, namespace,
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
    parser.add_argument('--policy-checkpoint', type=str,
                        help='Enable full routing, memory and critic inference with this Stage 3 policy')
    parser.add_argument('--memory-checkpoint', type=str, help='Override episodic checkpoint from policy')
    parser.add_argument('--semantic-store', type=str, help='Override semantic store from policy')
    parser.add_argument('--critic-checkpoint', type=str, help='Override critic checkpoint from policy')
    parser.add_argument('--memory-dir', type=str, help='Persist runtime memory between runs')
    parser.add_argument('--namespace', type=str, help='World memory namespace (random by default)')
    parser.add_argument('--route-policy', choices=['learned', 'always', 'never', 'bypass'],
                        help='Override routing policy for an ablation')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    engine = EvoInferenceEngine(
        args.checkpoint, device=args.device, quantize=args.quantize,
        policy_checkpoint=args.policy_checkpoint, memory_checkpoint=args.memory_checkpoint,
        semantic_store=args.semantic_store, critic_checkpoint=args.critic_checkpoint,
        memory_dir=args.memory_dir, route_policy=args.route_policy,
    )

    gen_kwargs = dict(
        max_tokens_per_tick=args.max_tokens_per_tick,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        namespace=args.namespace,
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
        if engine.full_engine is not None:
            stats = engine.full_engine.get_stats()
            print(f"Routing: {stats['bypasses']} bypasses, {stats['episodic_accesses']} episodic reads, "
                  f"{stats['semantic_accesses']} semantic reads, {stats['verifications']} verifications, "
                  f"{stats['abstentions']} abstentions", file=sys.stderr)

    finally:
        if out_file:
            out_file.close()
        engine.close()


if __name__ == '__main__':
    main()
