"""
Inference Script for HMST

Loads a trained model and generates text from prompts.
"""

import torch
import torch.nn.functional as F
from hmst.models import BaseMoEModel
from hmst.tokenizer import HMSTTokenizer
import os


def load_model(checkpoint_path, device=None):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        config: Model configuration
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    config = checkpoint['config']

    # Initialize model
    model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load tokenizer
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = HMSTTokenizer.load(tokenizer_path)
    else:
        print("Warning: Tokenizer not found in checkpoint directory, using default GPT-2 tokenizer")
        tokenizer = HMSTTokenizer()

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Trained for {checkpoint['epoch']} epochs")

    # Print validation metrics if available
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Validation perplexity: {checkpoint['val_ppl']:.2f}")

    return model, tokenizer, config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    device='cuda'
):
    """
    Generate text from a prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        device: Device

    Returns:
        generated_text: Generated text
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Generating up to {max_length} tokens...\n")
    print(f"Output: {prompt}", end='', flush=True)

    # Generate tokens one at a time
    for _ in range(max_length):
        # Forward pass
        output = model(input_tensor)
        logits = output['logits']

        # Get logits for last token
        next_token_logits = logits[0, -1, :] / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            min_top_k = top_k_values[-1]
            next_token_logits = torch.where(
                next_token_logits < min_top_k,
                torch.tensor(float('-inf'), device=next_token_logits.device),
                next_token_logits
            )

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            # Keep at least one token to avoid empty distribution
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.numel() > 1:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        # Decode and print token
        token_text = tokenizer.decode([next_token.item()], skip_special_tokens=False)
        print(token_text, end='', flush=True)

        # Stop at end-of-sequence
        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n")

    # Decode full sequence
    generated_ids = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def interactive_mode(model, tokenizer, device='cuda'):
    """
    Interactive text generation mode.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device
    """
    print("\n" + "="*80)
    print("HMST Interactive Generation Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Type a prompt to generate text")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'help' for options\n")

    max_length = 50
    temperature = 0.8
    top_p = 0.9
    top_k = 50

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
                print(f"  max_length: {max_length}")
                print(f"  temperature: {temperature}")
                print(f"  top_p: {top_p}")
                print(f"  top_k: {top_k}")
                print(f"\nTo change settings, type: set <param> <value>")
                print(f"Example: set max_length 100")
                continue

            if prompt.lower().startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param == 'max_length':
                        max_length = int(value)
                        print(f"Set max_length to {max_length}")
                    elif param == 'temperature':
                        temperature = float(value)
                        print(f"Set temperature to {temperature}")
                    elif param == 'top_p':
                        top_p = float(value)
                        print(f"Set top_p to {top_p}")
                    elif param == 'top_k':
                        top_k = int(value)
                        print(f"Set top_k to {top_k}")
                    else:
                        print(f"Unknown parameter: {param}")
                continue

            # Generate text
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HMST Inference')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, help='Prompt for generation (non-interactive)')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model, tokenizer, config = load_model(args.checkpoint, device=device)

    # Interactive or single generation
    if args.prompt:
        # Single generation
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device
        )
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, device=device)
