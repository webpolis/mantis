# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Status**: Research prototype with complete architecture implementation but no trained models.

HMST (Hierarchical Memory-State Transformer) is a novel LLM architecture design exploring hallucination mitigation and long-context memory through:
- Three-tier memory hierarchy (Attention → SSM → FAISS)
- RL-trainable meta-controller for dynamic routing
- Sparse MoE base model (designed for 12B total, 2B active parameters)
- Integrated critic model for hallucination detection

**Note**: System requires large-scale training ($750K-$1.5M in compute) to validate design hypotheses. Current implementation uses untrained/random weights.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v
python -m pytest tests/test_memory.py -v

# Run demo
python demo.py
```

### Training

**Two Training Modes:**

1. **Convenience Mode** (quick iteration): Auto-split validation from training data
2. **Production Mode** (reproducible): Pre-split datasets for consistent evaluation

#### RECOMMENDED: Pre-tokenize + Production Mode

Pre-tokenization is 5-10x faster and used by all major LLM projects (GPT, LLaMA, etc.):

```bash
# Step 1: Pre-tokenize your data (do this once)
python scripts/preprocess_data.py \
    --input /path/to/train.txt \
    --output data/tokenized/train

# Step 2A: PRODUCTION MODE - Pre-split for reproducibility
python scripts/split_dataset.py  # creates train_split and val directories
python train.py data/tokenized/train_split \
    --pretokenized \
    --val-file data/tokenized/val \
    --output-dir checkpoints/my_run

# Step 2B: CONVENIENCE MODE - Auto-split during training
python train.py data/tokenized/train \
    --pretokenized \
    --val-split 0.1 \
    --output-dir checkpoints/my_run
```

**Alternative: On-the-Fly Tokenization** (slower, for small datasets):

```bash
# Basic single-GPU training with auto-split validation
python train.py data/train.txt --val-split 0.1

# Production mode with pre-split validation
python train.py data/train.txt --val-file data/val.txt --output-dir checkpoints/my_run

# Use existing tokenizer from previous checkpoint
python train.py data/train.txt \
    --tokenizer-path checkpoints/improved_train/tokenizer \
    --val-split 0.1

# Control steps per epoch (useful for large datasets or quick iterations)
python train.py data/train.txt --steps-per-epoch 1000 --epochs 50 --val-split 0.1

# Multi-GPU training (auto-detected)
python train.py data/train.txt --batch-size 4 --val-split 0.1

# Multi-GPU with specific GPUs only (useful for shared systems)
python train.py data/train.txt --gpu-ids 0 2 --val-split 0.1

# Mixed VRAM GPUs (e.g., 12GB + 6GB): use gradient accumulation
# Effective batch size per GPU: 2 × 4 = 8
# Total effective batch size: 8 × N_GPUs
python train.py data/train.txt \
    --gradient-accumulation-steps 4 \
    --batch-size 2 \
    --mixed-precision \
    --val-split 0.1

# Train larger model with mixed precision
python train.py data/train.txt --model-size small --mixed-precision --val-split 0.1

# Memory-optimized for small model (~1B) on limited VRAM (12GB)
python train.py data/train.txt \
    --model-size small \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --mixed-precision \
    --gradient-checkpointing \
    --use-8bit-optimizer \
    --val-split 0.1

# Full example with all features (convenience mode)
python train.py data/train.txt \
    --val-split 0.1 \
    --output-dir checkpoints/full_run \
    --tokenizer-path checkpoints/improved_train/tokenizer \
    --model-size tiny \
    --epochs 20 \
    --batch-size 8 \
    --learning-rate 3e-4 \
    --steps-per-epoch 1000 \
    --eval-every 500 \
    --patience 5 \
    --save-every 5 \
    --mixed-precision
```

**Validation Split Options**:
- `--val-split 0.1`: Auto-split 10% for validation (convenience, reshuffles each run)
- `--val-file data/val.txt`: Use pre-split validation (production, reproducible)
- Cannot use both `--val-split` and `--val-file` together

**Tokenizer Management**:
- First run creates new tokenizer, saved to `<output-dir>/tokenizer`
- Reuse existing tokenizer: `--tokenizer-path checkpoints/run1/tokenizer`
- Ensures vocabulary consistency across training runs

**Model Sizes**:
- `micro`: ~10M parameters (ultra-fast testing, TinyStories dataset)
- `tiny`: ~100M parameters (testing/development)
- `small`: ~1B parameters (experimentation)
- `base`: ~12B parameters (production)

**Gradient Accumulation & Mixed VRAM GPUs**:
- Use `--gradient-accumulation-steps N` to accumulate gradients over N micro-batches before optimizer update
- Effective batch size = `batch_size × gradient_accumulation_steps × num_gpus`
- Essential for mixed VRAM setups (e.g., 12GB + 6GB GPUs)
- Learning rate schedule automatically adjusts for accumulation steps
- Validation timing based on optimizer steps (not batch count)
- Example: `--batch-size 2 --gradient-accumulation-steps 4` → effective batch size of 8 per GPU

**GPU Selection**:
- By default: Uses all available GPUs automatically (via Accelerate)
- `--gpu-ids 0`: Use only GPU 0
- `--gpu-ids 0 2`: Use specific GPUs (e.g., GPU 0 and GPU 2)
- Accelerate handles mixed architectures (e.g., Turing + Ampere) automatically
- For mixed VRAM, use gradient accumulation

**Memory Optimizations** (for large models on limited VRAM):
- `--gradient-checkpointing`: Trades 30% compute for 40% memory savings
- `--use-8bit-optimizer`: Cuts optimizer memory ~50% (requires `pip install bitsandbytes`)
- Combine with `--mixed-precision`, small `--batch-size`, and `--gradient-accumulation-steps`
- Example: Small model (~1B) fits in 12GB with all optimizations enabled

**Stage 3: RL Training** (meta-controller optimization):
```bash
python -m training.rl_train \
    --checkpoint checkpoints/pretrain/final.pt \
    --episodes 50000
```

Note: Stage 2 (memory fine-tuning) is not yet implemented.

### Inference

Use `inference.py` for production text generation with trained models. Supports interactive, single-prompt, and batch modes.

```bash
# Interactive mode (default)
python inference.py checkpoints/train/best_model.pt

# Single prompt
python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time"

# Batch generation from file
python inference.py checkpoints/train/best_model.pt \
    --input prompts.txt \
    --output results.txt

# Greedy decoding (deterministic, temperature=0)
python inference.py checkpoints/train/best_model.pt \
    --prompt "The capital of France is" \
    --temperature 0

# More creative generation (higher temperature)
python inference.py checkpoints/train/best_model.pt \
    --prompt "Write a story:" \
    --temperature 1.2 \
    --max-length 200

# Control sampling
python inference.py checkpoints/train/best_model.pt \
    --prompt "Hello world" \
    --top-p 0.95 \
    --top-k 40 \
    --max-length 100
```

**Generation Parameters**:
- `--max-length`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature, 0=greedy (default: 0.8)
- `--top-p`: Nucleus sampling threshold (default: 0.9)
- `--top-k`: Top-k sampling threshold (default: 50)
- `--output`: Save results to file

**Interactive Commands**:
- Type a prompt to generate text
- `set <param> <value>` - Change generation parameter (e.g., `set temperature 1.0`)
- `stats` - Show performance statistics
- `help` - Show current settings
- `quit` or `exit` - Exit

**Performance Metrics**: The script reports tokens/sec, latency, and time-to-first-token for each generation.

**Note**: For full HMST system inference with memory and routing, use the `HMSTInferenceEngine` in `hmst/inference/engine.py` (research-oriented, requires all components trained).

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .
```

## Architecture Overview

### Core Components

1. **BaseMoEModel** (`models/base_moe.py`): Sparse MoE transformer with 8 experts and top-2 routing. Each expert processes ~1/4 of the model capacity.

2. **MetaController** (`models/meta_controller.py`): Lightweight 6-layer transformer that makes 5 routing decisions per query:
   - Early exit (skip deep processing)
   - Episodic memory access
   - Semantic memory retrieval
   - Expert selection for MoE
   - Verification trigger for critic

3. **CriticModel** (`models/critic.py`): 1-2B parameter verification model that detects hallucinations by checking logical consistency and factual accuracy.

4. **EpisodicMemorySSM** (`models/ssm.py`): Mamba-based selective state space model that compresses 8K tokens into 256-dimensional state vectors.

### Memory Systems

1. **EpisodicMemory** (`memory/episodic.py`): Recent interaction buffer (L2 cache). Stores up to `max_entries` of recent context, compressed via SSM.

2. **SemanticMemory** (`memory/semantic.py`): FAISS-based long-term storage (L3 cache). Uses IVF/PQ indexing for efficient nearest-neighbor search over 1M+ entries.

3. **MemoryConsolidator** (`memory/consolidation.py`): Background process that transfers important memories from episodic to semantic storage based on importance scoring.

### Inference

**HMSTInferenceEngine** (`inference/engine.py`): Main orchestration system. For each query:
1. Encodes query with base model
2. Consults meta-controller for routing decisions
3. Optionally retrieves from episodic/semantic memory
4. Generates response through appropriate MoE experts
5. Optionally verifies with critic model

### Configuration

**HMSTConfig** (`configs/model_config.py`): Centralized configuration with three presets:
- `get_small_config()`: 1B parameters (demo/testing)
- `get_base_config()`: 12B parameters (production)
- `get_large_config()`: 30B parameters (research)

## Important Implementation Details

### Dimension Compatibility
- Base model embeddings: `d_model` (typically 2048)
- Episodic SSM state: `d_state` (typically 256)
- Semantic memory: `dimension` (typically 1024)

When querying episodic memory, embeddings must be projected from `d_model` to `d_state` space. This projection is handled in `episodic.py:retrieve()`.

### FAISS Index Training
Semantic memory uses IVF (Inverted File Index) which requires training on actual embeddings before use. The index is trained lazily on the first 10K+ additions. After training, all cached embeddings are re-added to the trained index.

### RL Training
The meta-controller is trained with PPO (Proximal Policy Optimization) using a multi-objective reward:
```
Reward = 1.0·Accuracy - 0.3·Latency - 0.2·Compute + 0.5·Calibration
```

State representation is created by encoding the query with the base model and computing a state summary via `StateSummaryEncoder`.

### Tokenization
Current implementation includes a BPE tokenizer but falls back to placeholder tokenization (random tokens) when no tokenizer is provided. The inference engine will generate meaningless outputs without a properly trained tokenizer. For production, ensure a tokenizer is trained during the initial training run or loaded from an existing checkpoint.

## Known Limitations

1. **No trained models**: All models use random initialization; requires large-scale training
2. **No evaluation framework**: `evaluation/` directory is empty; no benchmark code exists
3. **Placeholder tokenizer**: Falls back to random tokens when no tokenizer provided
4. **MoE expert routing**: Uses nested loops; needs vectorization for production speed
5. **SSM sequential scan**: Python loop implementation; requires parallel scan for GPU efficiency
6. **FAISS index rebuilding**: Full rebuild on deletion; consider using `remove_ids()`
7. **Memory consolidation**: Placeholder summary/encoding functions need base model integration
8. **Stage 2 training**: Memory fine-tuning script not implemented

## Module Import Pattern

```python
from hmst import (
    HMSTInferenceEngine,
    get_micro_config,
    get_tiny_config,
    get_small_config,
    get_base_config,
    get_large_config,
    BaseMoEModel,
    MetaController,
    CriticModel,
    EpisodicMemorySSM,
    EpisodicMemory,
    SemanticMemory
)
from hmst.models.meta_controller import StateSummaryEncoder
```

## Design Goals (Unvalidated)

The architecture is designed to achieve these theoretical targets (not yet measured):
- Hallucination reduction through critic-based verification
- Context scaling via hierarchical memory (8K attention + RAG-style retrieval from 1M+ entry database)
- Efficiency improvements through sparse expert activation (top-2 of 8 experts)
- Latency reduction via early-exit mechanisms

**Validation Requirements**:
- Large-scale training: 2-5T tokens, 500K-1M A100 GPU-hours, ~$1M cost
- Evaluation framework: MMLU, TruthfulQA, HumanEval, GSM8K, custom hallucination metrics
- Baseline comparisons: Run same tests on Llama-2, Mistral, etc.

These are architectural design goals, not proven capabilities. The `evaluation/` directory is currently empty.
