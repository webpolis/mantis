"""
Simple causal mask verification test (no external dependencies).
"""

import sys
import torch
import torch.nn as nn

# Import only what we need
sys.path.insert(0, '/home/nico/dev/projects/hmst')
from hmst.models.base_moe import BaseMoEModel


def test_causal_mask():
    """
    Test that modifying future tokens doesn't affect past predictions.
    """
    print("=" * 80)
    print("CAUSAL MASK DIAGNOSTIC TEST")
    print("=" * 80)

    # Initialize small model for speed
    print("\n1. Initializing model...")
    model = BaseMoEModel(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        n_experts=4,
        max_seq_len=128
    )
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {num_params/1e6:.2f}M parameters")

    # Create test inputs
    print("\n2. Creating test sequences...")
    seq_len = 10
    input_1 = torch.randint(0, 1000, (1, seq_len))
    input_2 = input_1.clone()

    # Modify FUTURE token (should not affect past predictions)
    modify_pos = 7
    test_pos = 3  # Earlier position
    input_2[0, modify_pos] = 999  # Change token at position 7

    print(f"   Sequence 1: {input_1[0, :8].tolist()}...")
    print(f"   Sequence 2: {input_2[0, :8].tolist()}... (modified position {modify_pos})")

    # Run forward passes
    print(f"\n3. Testing prediction at position {test_pos}...")
    with torch.no_grad():
        out_1 = model(input_1)['logits']
        out_2 = model(input_2)['logits']

    # Compare logits at early position
    logits_1 = out_1[0, test_pos, :]
    logits_2 = out_2[0, test_pos, :]

    max_diff = (logits_1 - logits_2).abs().max().item()
    mean_diff = (logits_1 - logits_2).abs().mean().item()

    print(f"   Max difference:  {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")

    # Verify causality
    print("\n4. RESULT:")
    threshold = 1e-5
    if max_diff < threshold:
        print(f"   ✅ SUCCESS: Future tokens (pos {modify_pos}) do not affect past predictions (pos {test_pos})")
        print(f"   ✅ Causal mask is working correctly!")
        return True
    else:
        print(f"   ❌ FAILURE: Information is leaking from future to past!")
        print(f"   ❌ Causal mask is NOT working correctly!")
        print(f"   ❌ Expected diff < {threshold:.2e}, got {max_diff:.2e}")
        return False


if __name__ == "__main__":
    success = test_causal_mask()
    print("\n" + "=" * 80)
    sys.exit(0 if success else 1)
