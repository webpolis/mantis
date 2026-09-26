"""
Production Inference Script for MANTIS

Features:
- Load trained models from checkpoints
- Interactive and batch generation modes
- Multiple sampling strategies (greedy, top-k, top-p)
- Streaming token-by-token output
- Performance metrics (tokens/sec, latency)
- Quantization support (INT8 on CPU, FP16 on GPU)
- Clean CLI interface
- Proper error handling

Usage:
    # Interactive mode
    python inference.py checkpoints/train/best_model.pt

    # Single prompt
    python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time"

    # Batch from file
    python inference.py checkpoints/train/best_model.pt --input prompts.txt --output results.txt

    # Greedy decoding (deterministic)
    python inference.py checkpoints/train/best_model.pt --prompt "Hello" --temperature 0

    # INT8 dynamic quantization (CPU only)
    python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize int8

    # FP16 inference (GPU only, 2x faster)
    python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize float16
"""

import os

import torch
import warnings

# Disable TF32 for compatibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Suppress cuBLAS recovery warnings (these are expected when workaround is applied)
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

from mantis.inference.generation import generate_tokens
from mantis.utils.checkpoints import load_base_model


class InferenceEngine:
    """Production-ready inference engine for MANTIS models."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        quantize: Optional[str] = None
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto)
            quantize: Quantization mode ('int8', 'float16', or None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantize = quantize

        if quantize == 'int8' and self.device != 'cpu':
            print("INT8 dynamic quantization only has CPU kernels; running on CPU")
            self.device = 'cpu'

        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        if quantize:
            print(f"Quantization: {quantize}")

        try:
            self.model, self.tokenizer, checkpoint = load_base_model(checkpoint_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        if self.quantize == 'int8':
            print("Applying INT8 dynamic quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✓ Model quantized to INT8")
        elif self.quantize == 'float16':
            if self.device == 'cuda':
                self.model = self.model.half()
                print("✓ Model converted to FP16 (~50% memory)")
            else:
                print("Warning: FP16 only supported on CUDA, skipping quantization")

        self.banned_ids = self.tokenizer.non_generable_ids

        param_counts = self.model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M parameters ({param_counts['active'] / 1e6:.2f}M active)")
        print(f"Vocabulary: {len(self.tokenizer):,} tokens")
        print(f"Context window: {self.model.max_seq_len} tokens")

        if 'epoch' in checkpoint:
            print(f"Trained: {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"Validation perplexity: {math.exp(min(checkpoint['val_loss'], 100)):.2f}")

        # Statistics
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'num_generations': 0
        }

    def _tokens(self, prompt: str, max_length: int, temperature: float, top_p: float, top_k: int):
        """Encode the prompt and stream (token_id, log_prob) pairs until EOS."""
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("Prompt is empty")
        return input_ids, generate_tokens(
            self.model, input_ids, max_length,
            temperature=temperature, top_k=top_k, top_p=top_p,
            banned_ids=self.banned_ids, stop_ids=[self.tokenizer.eos_token_id],
        )

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = True
    ) -> Dict:
        """
        Generate text from prompt.

        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, >1 = more random)
            top_p: Nucleus sampling threshold (0-1)
            top_k: Top-k sampling threshold
            stop_sequences: List of sequences that stop generation
            stream: Stream output token-by-token

        Returns:
            Dict with generated text, tokens, and metrics
        """
        input_ids, token_stream = self._tokens(prompt, max_length, temperature, top_p, top_k)

        start_time = time.time()
        first_token_time = None

        if stream:
            print(f"\nPrompt: {prompt}")
            print(f"Output: {prompt}", end='', flush=True)
        decoder = self.tokenizer.stream_decoder()

        generated_tokens = []
        for token, _ in token_stream:
            if first_token_time is None:
                first_token_time = time.time()
            generated_tokens.append(token)

            if stream:
                print(decoder.push(token), end='', flush=True)

            if stop_sequences:
                current_text = self.tokenizer.decode(generated_tokens)
                if any(stop_seq in current_text for stop_seq in stop_sequences):
                    break

        total_time = time.time() - start_time
        ttft = first_token_time - start_time if first_token_time else 0.0

        if stream:
            print("\n")

        generated_text = self.tokenizer.decode(generated_tokens)
        num_tokens = len(generated_tokens)
        tokens_per_sec = num_tokens / total_time if total_time > 0 else 0

        self.stats['total_tokens'] += num_tokens
        self.stats['total_time'] += total_time
        self.stats['num_generations'] += 1

        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': prompt + generated_text,
            'num_tokens': num_tokens,
            'prompt_tokens': len(input_ids),
            'total_tokens': len(input_ids) + num_tokens,
            'time_seconds': total_time,
            'ttft_seconds': ttft,
            'tokens_per_second': tokens_per_sec
        }

    def generate_streaming(
        self,
        prompt: str,
        max_length: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        """Generator yielding decoded text one token at a time (EOS is not yielded)."""
        _, token_stream = self._tokens(prompt, max_length, temperature, top_p, top_k)
        decoder = self.tokenizer.stream_decoder()
        for token, _ in token_stream:
            if token == self.tokenizer.eos_token_id:
                break
            text = decoder.push(token)
            if text:
                yield text

    def print_stats(self):
        """Print aggregated statistics."""
        if self.stats['num_generations'] == 0:
            print("No generations yet")
            return

        avg_tokens_per_sec = self.stats['total_tokens'] / self.stats['total_time']
        avg_time = self.stats['total_time'] / self.stats['num_generations']

        print(f"\n{'='*60}")
        print("Generation Statistics")
        print(f"{'='*60}")
        print(f"Total generations: {self.stats['num_generations']}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Average tokens/sec: {avg_tokens_per_sec:.2f}")
        print(f"Average time per generation: {avg_time:.2f}s")
        print(f"{'='*60}\n")


def interactive_mode(engine: InferenceEngine, args):
    """Interactive generation mode."""
    print("\n" + "="*80)
    print("MANTIS Interactive Generation")
    print("="*80)
    print("\nCommands:")
    print("  Type a prompt to generate text")
    print("  'quit' or 'exit' - Exit")
    print("  'help' - Show current settings")
    print("  'set <param> <value>' - Change generation parameter")
    print("  'stats' - Show performance statistics")
    print()

    # Default settings
    settings = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k
    }

    while True:
        try:
            prompt = input("\nPrompt > ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'help':
                print(f"\nCurrent settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print(f"\nTo change: set <param> <value>")
                print(f"Example: set max_length 100")
                continue

            if prompt.lower() == 'stats':
                engine.print_stats()
                continue

            if prompt.lower().startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param in settings:
                        try:
                            if param == 'max_length' or param == 'top_k':
                                settings[param] = int(value)
                            else:
                                settings[param] = float(value)
                            print(f"✓ Set {param} = {settings[param]}")
                        except ValueError:
                            print(f"Error: Invalid value for {param}")
                    else:
                        print(f"Error: Unknown parameter '{param}'")
                        print(f"Available: {', '.join(settings.keys())}")
                continue

            # Generate
            result = engine.generate(
                prompt=prompt,
                max_length=settings['max_length'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                top_k=settings['top_k'],
                stream=True
            )

            # Print metrics
            print(f"\nMetrics: {result['num_tokens']} tokens, "
                  f"{result['time_seconds']:.2f}s, "
                  f"{result['tokens_per_second']:.1f} tok/s")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(engine: InferenceEngine, args):
    """Batch generation from file."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Read prompts
    print(f"Reading prompts from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(prompts)} prompts...")

    # Generate
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating...")
        result = engine.generate(
            prompt=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.verbose
        )
        results.append(result)

        if not args.verbose:
            print(f"✓ {result['num_tokens']} tokens in {result['time_seconds']:.2f}s")

    # Save outputs
    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated: {result['generated_text']}\n")
                f.write(f"Tokens: {result['num_tokens']}, Time: {result['time_seconds']:.2f}s\n")
                f.write("-" * 80 + "\n")

    # Print summary
    engine.print_stats()


def single_prompt_mode(engine: InferenceEngine, args):
    """Generate from single prompt."""
    result = engine.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=True
    )

    # Print metrics
    print(f"\n{'='*60}")
    print("Generation Metrics")
    print(f"{'='*60}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Generated tokens: {result['num_tokens']}")
    print(f"Total time: {result['time_seconds']:.3f}s")
    print(f"Time to first token: {result['ttft_seconds']:.3f}s")
    print(f"Throughput: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"{'='*60}\n")

    # Save if requested
    if args.output:
        print(f"Saving to: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result['full_text'])


def main():
    parser = argparse.ArgumentParser(
        description='MANTIS Production Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python inference.py checkpoints/train/best_model.pt

  # Single prompt
  python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time"

  # Batch from file
  python inference.py checkpoints/train/best_model.pt --input prompts.txt --output results.txt

  # Greedy decoding (deterministic)
  python inference.py checkpoints/train/best_model.pt --prompt "Hello" --temperature 0

  # INT8 dynamic quantization (CPU only)
  python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize int8

  # FP16 inference (GPU only, 2x faster)
  python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize float16
        """
    )

    # Required
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    # Generation modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--prompt', type=str, help='Single prompt for generation')
    mode_group.add_argument('--input', type=str, help='Input file with prompts (one per line)')

    # Generation parameters
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum tokens to generate (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature, 0=greedy (default: 0.8)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling threshold (default: 0.9)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling threshold (default: 50)')

    # Output
    parser.add_argument('--output', type=str, help='Output file to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output in batch mode')

    # System
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--quantize', type=str, choices=['int8', 'float16'],
                        help='Quantization mode: int8 (CPU only) or float16 (GPU only)')

    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize engine
    try:
        engine = InferenceEngine(args.checkpoint, device=args.device, quantize=args.quantize)
    except Exception as e:
        print(f"Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Select mode
    if args.prompt:
        # Single prompt mode
        single_prompt_mode(engine, args)
    elif args.input:
        # Batch mode
        batch_mode(engine, args)
    else:
        # Interactive mode (default)
        interactive_mode(engine, args)


if __name__ == '__main__':
    main()
