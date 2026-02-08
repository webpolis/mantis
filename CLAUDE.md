# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MANTIS (Metacognitive Adaptive Network with Tiered Inference Strategies) is a research prototype LLM architecture exploring hallucination mitigation and long-context memory through metacognitive routing and hierarchical memory systems. The architecture consists of:

- **Base MoE Model**: Sparse Mixture-of-Experts transformer (~12B total, ~2B active parameters)
- **Three-tier memory**: Attention (8K) → Episodic SSM → Semantic FAISS
- **Meta-controller**: RL-trainable routing with 5 decision gates
- **Critic model**: Integrated hallucination detection

**Status**: Complete architecture implementation but no trained models.

## Training Commands

### Stage 1: Base MoE Pre-training (REQUIRED)

```bash
# Basic training with HuggingFace streaming dataset
python train.py --stage 1 \
    --hf-dataset roneneldan/TinyStories \
    --hf-val-split validation \
    --streaming \
    --steps-per-epoch 1000

# Local file with auto-split validation (convenient)
python train.py --stage 1 data/train.txt --val-split 0.1

# Production: Pre-tokenized dataset with separate validation
python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
python scripts/split_dataset.py
python train.py --stage 1 data/tokenized/train_split \
    --pretokenized \
    --val-file data/tokenized/val

# Resume training
python train.py --stage 1 data/train.txt \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --val-split 0.1

# Multi-GPU with memory optimizations
python train.py --stage 1 data/train.txt \
    --model-size small \
    --mixed-precision \
    --gradient-checkpointing \
    --use-8bit-optimizer \
    --val-split 0.1
```

### Stage 2: Memory Fine-tuning (OPTIONAL)

```bash
python train.py --stage 2 \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --epochs 5
```

### Stage 3: RL Training (OPTIONAL)

```bash
python train.py --stage 3 \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --rl-episodes 50000
```

## Inference Commands

```bash
# Interactive mode
python inference.py checkpoints/stage1/best_model.pt

# Single prompt
python inference.py checkpoints/stage1/best_model.pt \
    --prompt "Once upon a time"

# Greedy decoding (deterministic)
python inference.py checkpoints/stage1/best_model.pt \
    --prompt "Hello" \
    --temperature 0

# Quantized inference (2-4x faster)
python inference.py checkpoints/stage1/best_model.pt \
    --prompt "Hello" \
    --quantize int8
```

### RTX 3060 cuBLAS Bug Workaround

If you encounter `CUBLAS_STATUS_NOT_INITIALIZED` errors during inference:

```bash
export CUBLAS_WORKSPACE_CONFIG=:0:0
export TORCH_BLAS_PREFER_CUBLASLT=0
python inference.py checkpoints/stage1/best_model.pt --prompt "Hello"
```

This bug affects RTX 30xx/40xx series and A-series Ampere GPUs at sequence length ≥5 with large vocabulary matrices (128K tokens). The workaround is automatically applied in `train.py` and `inference.py`.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_models.py -v

# With coverage
python -m pytest tests/ --cov=mantis --cov-report=html
```

## Evaluation

```bash
# Test with demo dataset
python scripts/run_eval.py checkpoints/stage1/best_model.pt \
    --tokenizer checkpoints/stage1/tokenizer \
    --all --demo

# Run specific benchmarks
python scripts/run_eval.py checkpoints/stage1/best_model.pt \
    --tokenizer checkpoints/stage1/tokenizer \
    --benchmarks mmlu truthfulqa \
    --output results.json
```

## Architecture Overview

### Project Structure

```
mantis/
├── models/              # Neural network architectures
│   ├── base_moe.py      # Base MoE model with sparse experts
│   ├── meta_controller.py  # RL-trainable routing controller
│   ├── critic.py        # Hallucination detection model
│   └── ssm.py           # State-space model for episodic memory
├── memory/              # Memory systems
│   ├── episodic.py      # SSM-based short-term memory (8K tokens)
│   ├── semantic.py      # FAISS-based long-term memory (1M+ entries)
│   └── consolidation.py # Memory transfer logic
├── training/            # Training pipelines
│   ├── pretrain.py      # Stage 1: Base MoE pre-training
│   ├── memory_train.py  # Stage 2: Memory fine-tuning
│   └── rl_train.py      # Stage 3: RL meta-controller training
├── inference/           # Generation engine
│   └── engine.py        # Dynamic routing and generation
├── configs/             # Configuration management
│   └── model_config.py  # Presets: micro/tiny/small/base
└── tokenizer.py         # MANTISTokenizer (GPT-2 BPE wrapper)

evaluation/              # Evaluation harness
├── benchmarks.py        # MMLU, TruthfulQA, HumanEval, GSM8K
└── metrics.py           # Accuracy, F1, hallucination rate

scripts/                 # Utility scripts
├── preprocess_data.py   # Pre-tokenize datasets (5-10x faster)
├── split_dataset.py     # Split train/val for reproducibility
└── run_eval.py          # Run benchmark evaluations
```

### Key Components

#### BaseMoEModel (`mantis/models/base_moe.py`)

Sparse MoE transformer with:
- 8 experts, top-2 routing
- Load balancing loss
- Scales from 100M to 12B parameters
- Standard transformer backbone with rotary positional embeddings

**Important**: The model uses `top_k=2` sparse routing by default. When modifying expert selection, ensure load balancing loss is maintained to prevent expert collapse.

**Meta-controller integration**: When `expert_weights` are provided by the meta-controller, they are applied as additive bias to the learned gate logits (not as replacement routing). This allows the meta-controller to influence expert selection while preserving learned routing patterns.

#### MetaController (`mantis/models/meta_controller.py`)

RL-trainable routing controller with 5 gates:
1. **Early exit**: Skip processing for simple queries
2. **Episodic memory**: Access recent context (8K tokens)
3. **Semantic memory**: Retrieve long-term facts (1M+ entries)
4. **Expert selection**: Additive bias to MoE routing (raw logits, not probabilities)
5. **Verification**: Trigger critic model

Trained with PPO in Stage 3. Gates 1-3 and 5 output continuous values [0,1] thresholded for binary decisions. Gate 4 outputs raw logits applied as bias to learned routing.

#### Memory Systems

**Episodic Memory** (`mantis/memory/episodic.py`):
- Mamba SSM using mamba-ssm library (CUDA-optimized)
- 8K token window
- L2 cache in memory hierarchy
- Thread-safe operations with locking for concurrent access

**Semantic Memory** (`mantis/memory/semantic.py`):
- FAISS vector database with IndexIDMap for efficient deletion
- 1M+ entries capacity
- L3 cache in memory hierarchy
- Deferred IVF index rebuild (only when >20% entries stale)
- **WARNING**: Stores embeddings in RAM (12GB+ for 1M entries). Limit to ~100K entries on 16GB RAM systems.

**Consolidation** (`mantis/memory/consolidation.py`):
- Background transfer from episodic → semantic
- Uses importance scoring and decay
- Encodes facts using base model embeddings (no longer placeholder)

#### Critic Model (`mantis/models/critic.py`)

1-2B parameter verification model for hallucination detection via consistency checking. Used when meta-controller triggers verification gate.

### Model Configuration

Model sizes are defined in `mantis/configs/model_config.py`:

- **micro**: ~10M parameters (dense, not MoE) - ultra-fast testing
- **tiny**: ~100M parameters (4 experts) - development/debugging
- **small**: ~1B parameters (4 experts) - experimentation
- **base**: ~12B parameters (8 experts) - production target

**Vocabulary**: All models use 50304 tokens (GPT-2 standard vocabulary size).

**Critical alignment requirements** (validated in `MANTISConfig.__post_init__`):
- `MetaControllerConfig.d_model` must match `BaseMoEConfig.d_model`
- `MetaControllerConfig.n_experts` must match `BaseMoEConfig.n_experts`
- `SemanticMemoryConfig.dimension` must match `BaseMoEConfig.d_model`

When adding new model sizes, ensure these dimensions are synchronized. Config mismatches will raise assertions at initialization.

### Training Pipeline

MANTIS uses a 3-stage training pipeline:

1. **Stage 1 (REQUIRED)**: Base MoE pre-training
   - Standard next-token prediction
   - Load balancing loss for expert utilization
   - Output: Functional LLM ready for text generation

2. **Stage 2 (OPTIONAL)**: Memory fine-tuning
   - Trains episodic + semantic memory systems
   - Enables context beyond 8K tokens
   - Requires Stage 1 checkpoint

3. **Stage 3 (OPTIONAL)**: RL training
   - Trains meta-controller routing policy with PPO
   - Optimizes accuracy/latency/compute trade-offs
   - Requires Stage 1 (or Stage 2) checkpoint

Each stage builds on the previous. Training is implemented in `mantis/training/` with separate modules per stage.

## Development Guidelines

### Model Modifications

When modifying model architectures:

1. **Dimension alignment**: If changing `d_model` in BaseMoE, update MetaController and SemanticMemory configs
2. **Expert count**: If changing `n_experts`, update MetaController config
3. **Checkpoint format**: All checkpoints must include `config` dict for proper resumption
4. **Load balancing**: MoE changes must preserve load balancing loss to prevent expert collapse

### Training Modifications

The main training script (`train.py`) uses HuggingFace Accelerate for:
- Multi-GPU training (auto-detected)
- Mixed precision (FP16)
- Gradient accumulation
- DeepSpeed ZeRO-3 (optional)

**Critical**: When modifying training loop:
- Restore model state using `accelerator.unwrap_model()` before loading checkpoints
- Round `steps_per_epoch` to nearest multiple of `gradient_accumulation_steps` to prevent stale gradient leakage
- Validate tokenizer exists when resuming (`--tokenizer-path` required)
- Save checkpoints with `config` dict for proper resumption

### Dataset Handling

Three dataset modes supported:

1. **Raw text** (`TextDataset`): Tokenizes on-the-fly, slower but simple
2. **Pre-tokenized** (`PreTokenizedDataset`): 5-10x faster, requires preprocessing with `scripts/preprocess_data.py`
3. **HuggingFace streaming** (`HuggingFaceDataset`): No download needed, use with `--streaming` flag

**WARNING**: `TextDataset` loads all tokens into RAM (not true streaming). For files >10GB, use `--pretokenized` or `--hf-dataset --streaming`.

### Validation Split Modes

Two validation modes:

1. **Auto-split** (`--val-split 0.1`): Convenient for iteration, reshuffles each run
2. **Pre-split** (`--val-file data/val.txt`): Reproducible, recommended for production

Cannot combine both. HuggingFace datasets use `--hf-val-split` instead.

### Inference Engine

Full MANTIS inference requires all components:
- Base MoE model
- Meta-controller
- Episodic memory
- Semantic memory
- Critic model

For basic generation (Stage 1 only), use `inference.py` which provides simplified inference without memory systems.

Production inference engine is in `mantis/inference/engine.py` and coordinates:
1. Query encoding
2. Uncertainty estimation
3. Meta-controller routing
4. Memory retrieval (episodic/semantic)
5. Generation with expert routing
6. Optional critic verification

### KV Caching

KV caching is implemented in `BaseMoEModel` for efficient inference. When adding new attention mechanisms, ensure:
- Cache shape: `(batch, n_heads, seq_len, d_head)`
- Cache is optional (training doesn't use it)
- Cache grows incrementally during generation

### Known Issues

1. **Semantic Memory Scaling**: RAM-based storage limits to ~100K entries on 16GB systems. Disk-backed storage planned.

2. **TextDataset Memory**: Not true streaming, loads all tokens into RAM. Use `--pretokenized` or `--streaming` for large datasets.

3. **RTX 3060 cuBLAS Bug**: Ampere GPUs have kernel bug with large vocab matrices at seq_len ≥5. Automatic workaround applied in `train.py` and `inference.py`.

4. **mamba-ssm Dependency**: Requires CUDA-capable GPU for compilation and runtime. CPU-only systems cannot use episodic memory.

## Important Notes

- **No trained weights**: This is a research prototype with architecture only. Full training requires 500K-1M A100 GPU-hours.
- **Validation required**: Design claims (reduced hallucinations, extended context) are unvalidated and require large-scale training + evaluation.
- **True attention limited to 8K**: Not 1M despite claims. Memory systems extend context, but base attention is 8K max.
- Always use `--tokenizer-path` when resuming training to ensure vocabulary consistency.
- When using `--resume`, the model config is loaded from checkpoint, not CLI args (except vocab_size which syncs with tokenizer).
- Gradient accumulation steps should divide evenly into steps_per_epoch to prevent stale gradients leaking across epochs.
