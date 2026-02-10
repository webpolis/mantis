"""
Production Inference Script for MANTIS

Features:
- Load trained models from checkpoints
- Interactive and batch generation modes
- Multiple sampling strategies (greedy, top-k, top-p)
- Streaming token-by-token output
- Performance metrics (tokens/sec, latency)
- Quantization support (INT8, FP16) for faster inference
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

    # Quantized inference (2-4x faster)
    python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize int8

    # FP16 inference (GPU only, 2x faster)
    python inference.py checkpoints/train/best_model.pt --prompt "Hello" --quantize float16
"""

import os
import subprocess

# Detect and fix RTX 3060 cuBLAS bug
# RTX 3060 (and some other Ampere GPUs) have a cuBLASLt kernel bug with large matrices
# that causes CUBLAS_STATUS_NOT_INITIALIZED at specific sequence lengths
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        gpu_names = result.stdout.strip().split('\n')
        # Check for RTX 30xx series or A-series Ampere GPUs (known to have cuBLAS bug)
        # RTX 20xx (Turing) does NOT have this issue
        problematic_gpus = ['RTX 30', 'RTX 40', 'A4000', 'A5000', 'A6000']
        detected_buggy = [name for name in gpu_names if any(gpu in name for gpu in problematic_gpus)]

        if detected_buggy:
            # Apply cuBLAS workaround
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
            os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
            print(f"⚠️  Detected Ampere GPU with known cuBLAS bug: {', '.join(detected_buggy)}")
            print(f"✓  Applied cuBLAS workaround (forces legacy cuBLAS, slight performance impact)\n")
except Exception:
    # If nvidia-smi fails, silently continue (might be CPU-only or different setup)
    pass

import torch
import warnings

# Disable TF32 for compatibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Suppress cuBLAS recovery warnings (these are expected when workaround is applied)
warnings.filterwarnings('ignore', message='.*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*')

import torch.nn.functional as F
from mantis.models import BaseMoEModel
from mantis.tokenizer import MANTISTokenizer
from mantis.utils.checkpoints import compat_load
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List


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

        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        if quantize:
            print(f"Quantization: {quantize}")

        # Load checkpoint
        try:
            checkpoint = compat_load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Extract config
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing 'config' field")
        config = checkpoint['config']

        # Initialize model
        self.model = BaseMoEModel(
            vocab_size=config.base_moe.vocab_size,
            d_model=config.base_moe.d_model,
            n_layers=config.base_moe.n_layers,
            n_heads=config.base_moe.n_heads,
            d_ff=config.base_moe.d_ff,
            n_experts=config.base_moe.n_experts,
            top_k=config.base_moe.top_k,
            max_seq_len=config.base_moe.max_seq_len,
            dropout=config.base_moe.dropout
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Apply quantization if requested
        if self.quantize == 'int8':
            print("Applying INT8 dynamic quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✓ Model quantized to INT8 (2-4x faster, ~75% memory)")
        elif self.quantize == 'float16':
            if self.device == 'cuda':
                print("Converting to FP16...")
                self.model = self.model.half()
                print("✓ Model converted to FP16 (~50% memory)")
            else:
                print("Warning: FP16 only supported on CUDA, skipping quantization")

        # Load tokenizer
        checkpoint_dir = os.path.dirname(checkpoint_path)
        tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')

        if os.path.exists(tokenizer_path):
            self.tokenizer = MANTISTokenizer.load(tokenizer_path)
            print(f"Tokenizer loaded from: {tokenizer_path}")
        else:
            print("Warning: Tokenizer not found, using default tokenizer")
            self.tokenizer = MANTISTokenizer()

        # Print model info
        param_counts = self.model.count_parameters()
        print(f"Model: {param_counts['total'] / 1e6:.2f}M parameters ({param_counts['active'] / 1e6:.2f}M active)")
        print(f"Vocabulary: {len(self.tokenizer):,} tokens")

        if 'epoch' in checkpoint:
            print(f"Trained: {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"Validation perplexity: {checkpoint.get('val_ppl', 'N/A')}")

        # Statistics
        self.stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'num_generations': 0
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
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
            num_beams: Number of beams for beam search (1 = no beam search)
            stop_sequences: List of sequences that stop generation
            stream: Stream output token-by-token

        Returns:
            Dict with generated text, tokens, and metrics
        """
        self.model.eval()

        # Encode prompt
        try:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        except Exception as e:
            raise ValueError(f"Failed to tokenize prompt: {e}")

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        prompt_length = len(input_ids)

        # Start timing
        start_time = time.time()
        first_token_time = None

        # Initialize output
        if stream:
            print(f"\nPrompt: {prompt}")
            print(f"Output: {prompt}", end='', flush=True)

        generated_tokens = []

        # Check if greedy decoding
        is_greedy = (temperature == 0.0)
        if is_greedy:
            temperature = 1.0  # Avoid division by zero

        # Beam search not implemented yet
        if num_beams > 1:
            raise NotImplementedError("Beam search not yet implemented")

        # Generate tokens with KV-cache
        past_key_values = None
        for step in range(max_length):
            if past_key_values is None:
                model_input = input_tensor[:, -self.model.max_seq_len:]
            else:
                model_input = input_tensor[:, -1:]

            # Forward pass with KV-cache
            output = self.model(model_input, past_key_values=past_key_values, use_cache=True)
            logits = output['logits']
            past_key_values = output['past_key_values']

            # Truncate cache if it exceeds max_seq_len
            if past_key_values[0][0].size(1) >= self.model.max_seq_len:
                past_key_values = [
                    (k[:, -self.model.max_seq_len + 1:], v[:, -self.model.max_seq_len + 1:])
                    for k, v in past_key_values
                ]

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            if is_greedy:
                # Greedy: take argmax
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Sampling: apply top-k and top-p filtering
                next_token_logits = self._apply_sampling_filters(
                    next_token_logits, top_k=top_k, top_p=top_p
                )

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Record first token time
            if first_token_time is None:
                first_token_time = time.time()

            # Append to sequence
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())

            # Decode and stream
            if stream:
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=False)
                print(token_text, end='', flush=True)

            # Check stopping conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            if stop_sequences:
                # Check if any stop sequence is in generated text
                current_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                if any(stop_seq in current_text for stop_seq in stop_sequences):
                    break

        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else 0.0

        if stream:
            print("\n")

        # Decode full output
        generated_ids = input_tensor[0].tolist()
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Compute metrics
        num_tokens = len(generated_tokens)
        tokens_per_sec = num_tokens / total_time if total_time > 0 else 0

        # Update stats
        self.stats['total_tokens'] += num_tokens
        self.stats['total_time'] += total_time
        self.stats['num_generations'] += 1

        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'num_tokens': num_tokens,
            'prompt_tokens': prompt_length,
            'total_tokens': prompt_length + num_tokens,
            'time_seconds': total_time,
            'ttft_seconds': ttft,
            'tokens_per_second': tokens_per_sec
        }

    def _apply_sampling_filters(
        self,
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits."""

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[-1]
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.numel() > 1:
                # Shift right to keep at least one token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        return logits

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_length: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        """Generator yielding token strings one at a time.

        Same generation logic as generate() (KV-cache, top-k/top-p sampling)
        but yields instead of accumulating.
        """
        self.model.eval()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        is_greedy = (temperature == 0.0)
        if is_greedy:
            temperature = 1.0

        past_key_values = None
        for _ in range(max_length):
            if past_key_values is None:
                model_input = input_tensor[:, -self.model.max_seq_len:]
            else:
                model_input = input_tensor[:, -1:]

            output = self.model(model_input, past_key_values=past_key_values, use_cache=True)
            logits = output['logits']
            past_key_values = output['past_key_values']

            if past_key_values[0][0].size(1) >= self.model.max_seq_len:
                past_key_values = [
                    (k[:, -self.model.max_seq_len + 1:], v[:, -self.model.max_seq_len + 1:])
                    for k, v in past_key_values
                ]

            next_token_logits = logits[0, -1, :] / temperature

            if is_greedy:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = self._apply_sampling_filters(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                break

            yield self.tokenizer.decode([token_id], skip_special_tokens=False)

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
        num_beams=args.num_beams,
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

  # Quantized inference (2-4x faster)
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
    parser.add_argument('--num-beams', type=int, default=1,
                        help='Number of beams for beam search (default: 1, no beam search)')

    # Output
    parser.add_argument('--output', type=str, help='Output file to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output in batch mode')

    # System
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--quantize', type=str, choices=['int8', 'float16'],
                        help='Quantization mode: int8 (2-4x faster, CPU/GPU) or float16 (2x faster, GPU only)')

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
