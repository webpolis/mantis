"""Test multi-GPU setup"""
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Initialize all GPUs
print("\nInitializing CUDA contexts on all GPUs...")
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        dummy = torch.randn(100, 100, device=f'cuda:{i}')
        result = torch.mm(dummy, dummy)
        torch.cuda.synchronize()
        print(f"GPU {i} initialized successfully")

# Create simple model
print("\nCreating simple model...")
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512)
).cuda()

print(f"Model on device: {next(model.parameters()).device}")

# Wrap with DataParallel
if torch.cuda.device_count() > 1:
    print(f"\nWrapping with DataParallel...")
    model = nn.DataParallel(model)
    print("DataParallel created successfully")

# Test forward pass
print("\nTesting forward pass...")
try:
    x = torch.randn(8, 512).cuda()
    y = model(x)
    print(f"Forward pass successful! Output shape: {y.shape}")

    # Test backward pass
    print("Testing backward pass...")
    loss = y.sum()
    loss.backward()
    print("Backward pass successful!")

    print("\n✓ Multi-GPU test PASSED!")

except Exception as e:
    print(f"\n✗ Multi-GPU test FAILED: {e}")
    import traceback
    traceback.print_exc()
