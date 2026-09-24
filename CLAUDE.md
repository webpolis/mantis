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

### Stages 2-4 (OPTIONAL)

Each takes the Stage 1 model and an optional JSONL file (demo data without it). Stage 1-only flags are rejected.

```bash
# Stage 2: memory fine-tuning. JSONL: {"query", "context"}
python train.py --stage 2 data/memory.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --epochs 5 --output-dir checkpoints/stage2

# Stage 4: critic training. JSONL: {"query", "response", "facts"?, "label"}
python train.py --stage 4 data/critic.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --output-dir checkpoints/critic

# Stage 3: RL routing policy. JSONL: {"query", "answer"}. Each optional component enables its gate.
python train.py --stage 3 data/qa.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --memory-checkpoint checkpoints/stage2/memory_system_final.pt \
    --semantic-store checkpoints/stage2/semantic_memory \
    --critic-checkpoint checkpoints/critic/critic_best.pt \
    --rl-episodes 50000 --output-dir checkpoints/stage3
```

## Evolution Curriculum Training

Dedicated training pipeline for the evolution simulation with per-token loss weighting and curriculum mixing across complexity tiers.

```bash
# 1. Generate partitioned datasets (cap by epoch)
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch CAMBRIAN  --output data/evo_bio.txt --compact --workers 8
python scripts/gen_evo_dataset.py --worlds 5000 --max-epoch ECOSYSTEM --output data/evo_eco.txt --compact --workers 8 --enable-agents
python scripts/gen_evo_dataset.py --worlds 5000                       --output data/evo_intel.txt --compact --workers 8 --enable-agents

# 2. Train with curriculum (all 3 partitions)
python train_evo.py \
    --bio data/evo_bio.txt --eco data/evo_eco.txt --intel data/evo_intel.txt \
    --model-size tiny --seq-len 2048 --batch-size 8 \
    --steps-per-epoch 1000 --epochs 20 \
    --learning-rate 5e-4 --warmup-steps 2000 \
    --mixed-precision --val-split 0.1

# Single partition (bio only)
python train_evo.py --bio data/evo_bio.txt --schedule bio-only \
    --model-size micro --seq-len 256 --batch-size 4 \
    --steps-per-epoch 100 --epochs 5 --val-split 0.1

# Resume from checkpoint
python train_evo.py --bio data/evo_bio.txt \
    --resume checkpoints/evo_train/best_model.pt \
    --tokenizer-path checkpoints/evo_train/tokenizer \
    --steps-per-epoch 1000 --epochs 40 --val-split 0.1
```

**Key differences from `train.py`**:
- Uses `EvoWorldDataset` (world-boundary-aware chunking, never crosses `\n\n` boundaries)
- Per-token loss weights via `tokenizer.compute_loss_weights()`, computed once per whole world (protocol markers set weight for subsequent tokens); ignored labels get weight 0
- `--val-split` holds out whole worlds per partition
- `CurriculumDataset` is an endless stream; partitions are addressed by name, and each schedule entry sets token shares that follow training progress (micro-steps done / planned)
- `--steps-per-epoch` defines an epoch (required); the LR schedule and curriculum both follow it
- Three schedule presets: `default` (gradual shift), `linear`, `bio-only`

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

# INT8 dynamic quantization (always runs on CPU)
python inference.py checkpoints/stage1/best_model.pt \
    --prompt "Hello" \
    --quantize int8
```

All decode loops share `mantis/inference/generation.py`: KV cache, banned tokens (pad, bos, unk, reserved), and a context window of `max_seq_len`. When the cache fills, the latest half-window is re-encoded from scratch.

### Evolution Inference

Tick-by-tick generation of evolution simulation traces. Importable as a module for web apps.

```bash
# Generate a new world
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --seed 42 --max-ticks 100

# Continue from partial trace
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --continue trace.txt --max-ticks 50

# Generate from custom prompt
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --prompt "=EPOCH 1 1000 W0"
```

```python
# Python API (for web app integration)
from inference_evo import EvoInferenceEngine

engine = EvoInferenceEngine("checkpoints/evo_train/best_model.pt")
for tick in engine.generate_world(seed=42, temperature=0.7):
    send_to_client(tick)
```

### RTX 3060 cuBLAS Bug Workaround

If you encounter `CUBLAS_STATUS_NOT_INITIALIZED` errors during inference:

```bash
export CUBLAS_WORKSPACE_CONFIG=:0:0
export TORCH_BLAS_PREFER_CUBLASLT=0
python inference.py checkpoints/stage1/best_model.pt --prompt "Hello"
```

This bug affects RTX 30xx/40xx series and A-series Ampere GPUs at sequence length ≥5 with large vocabulary matrices (128K tokens). The workaround is automatically applied in `train.py`, `train_evo.py`, `inference.py`, and `inference_evo.py`.

## Evaluation

```bash
# Test with demo dataset
python scripts/run_eval.py checkpoints/stage1/best_model.pt --all --demo

# Run specific benchmarks (downloaded from the HuggingFace Hub)
python scripts/run_eval.py checkpoints/stage1/best_model.pt \
    --benchmarks mmlu truthfulqa --limit 500 \
    --output results.json

# Evaluate the full engine rebuilt from a Stage 3 policy
python scripts/run_eval.py checkpoints/stage1/best_model.pt --all \
    --policy-checkpoint checkpoints/stage3/meta_controller_rl.pt
```

Confidence is the geometric-mean probability of generated tokens. TruthfulQA counts a response as truthful when it is closer (token F1) to a true reference than to any false one. HumanEval runs generated code in a resource-limited subprocess, which is not a security sandbox.

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
│   ├── common.py        # Accelerator, optimizer, LR schedule shared by train.py/train_evo.py
│   ├── pretrain.py      # Standalone single-GPU Stage 1 trainer (train.py is the main path)
│   ├── memory_train.py  # Stage 2: Memory fine-tuning
│   ├── rl_train.py      # Stage 3: RL meta-controller training
│   └── critic_train.py  # Stage 4: Critic training
├── inference/           # Generation engine
│   ├── generation.py    # Shared decode loop (cache, window, sampling)
│   └── engine.py        # Dynamic routing and generation
├── simulation/          # Ecological simulator for evolution training data (see its CLAUDE.md)
├── configs/             # Configuration management
│   └── model_config.py  # Presets: micro/tiny/small/base
├── utils/checkpoints.py # Checkpoint schema, model/tokenizer loading
├── data.py              # Documents, EOS, packing, leak-free splits
└── tokenizer.py         # MANTISTokenizer (trie-based, 512 tokens, byte fallback)

evaluation/              # Evaluation harness
├── benchmarks.py        # MMLU, TruthfulQA, HumanEval, GSM8K
└── metrics.py           # Accuracy, F1, hallucination rate

train_evo.py             # Evolution curriculum training (weighted loss, partition mixing)
inference_evo.py         # Evolution tick-by-tick inference (importable for web apps)

scripts/                 # Utility scripts
├── preprocess_data.py   # Pre-tokenize datasets (5-10x faster)
├── gen_evo_dataset.py   # Generate evolution simulation traces (supports --max-epoch partitioning)
├── calc_seq_len.py      # Measure per-tick token counts and recommend --seq-len per partition
├── split_dataset.py     # Split train/val for reproducibility
└── run_eval.py          # Run benchmark evaluations

web/                     # Simulation playground: Flask + Socket.IO server, React/PixiJS client
```

### Key Components

#### BaseMoEModel (`mantis/models/base_moe.py`)

Sparse MoE transformer with:
- 8 experts, top-2 routing (dense feedforward when `n_experts == 1`, e.g. micro)
- Load balancing loss, weighted by `load_balance_weight`
- Scales from 10M (micro, dense) to 12B parameters
- Pre-norm transformer backbone with rotary positional embeddings and `scaled_dot_product_attention`
- `max_seq_len` is the attention window; new models set it to the training `--seq-len`

**Important**: The model uses `top_k=2` sparse routing by default. When modifying expert selection, ensure load balancing loss is maintained to prevent expert collapse.

**Meta-controller integration**: When `expert_weights` are provided by the meta-controller, they are applied as additive bias to the learned gate logits (not as replacement routing). This allows the meta-controller to influence expert selection while preserving learned routing patterns.

#### MetaController (`mantis/models/meta_controller.py`)

RL-trainable routing controller with 5 gates:
1. **Early exit**: Skip processing for simple queries
2. **Episodic memory**: Access recent context (8K tokens)
3. **Semantic memory**: Retrieve long-term facts (1M+ entries)
4. **Expert selection**: Additive bias to MoE routing (raw logits, not probabilities)
5. **Verification**: Trigger critic model

A residual MLP over the pooled query embedding and a state summary. Trained with PPO in Stage 3 as a stochastic policy: gates 1-3 and 5 are Bernoulli, and gate 4 is a Gaussian around its raw logits. Deterministic inference thresholds gate probabilities (`InferenceConfig.routing_threshold`) and uses the Gaussian mean as the routing bias. A gate whose component is missing stays 0 and adds nothing to the policy log-probability.

#### Memory Systems

**Episodic Memory** (`mantis/memory/episodic.py`):
- Mamba SSM using mamba-ssm library (CUDA-optimized)
- 8K token window
- L2 cache in memory hierarchy
- The engine stores every query and response; retrieval returns token IDs, which the engine decodes into context
- Entries have unique IDs; thread-safe operations with locking and snapshots

**Semantic Memory** (`mantis/memory/semantic.py`):
- FAISS vector database with stable IDs; eviction tombstones IDs and the index rebuilds when >20% are stale
- IVF serves exact search until 10K entries, then trains with a cluster count sized to the data
- Optional projection (trained in Stage 2) applied before L2-normalized indexing
- 1M+ entries capacity
- L3 cache in memory hierarchy
- **WARNING**: Stores embeddings in RAM (12GB+ for 1M entries). Limit to ~100K entries on 16GB RAM systems.

**Consolidation** (`mantis/memory/consolidation.py`):
- Background transfer from episodic → semantic
- Uses importance scoring and similarity grouping; stores each group's decoded text
- Encodes facts using base model embeddings, like queries
- Removes only entries whose semantic write succeeded

#### Critic Model (`mantis/models/critic.py`)

1-2B parameter verification model for hallucination detection via consistency checking. Trained in Stage 4. Used when meta-controller triggers verification gate; inputs are truncated to its `max_seq_len` budget.

### Model Configuration

Model sizes are defined in `mantis/configs/model_config.py`:

- **micro**: ~10M parameters (dense, not MoE) - ultra-fast testing
- **tiny**: ~100M parameters (4 experts) - development/debugging
- **small**: ~1B parameters (4 experts) - experimentation
- **base**: ~12B parameters (8 experts) - production target

`get_large_config()` (~30B, 16 experts) and `get_extmem_config()` (32K windows) exist too, but `--model-size` does not offer them.

**Vocabulary**: All models use 512 tokens (custom domain-specific trie tokenizer, synced at runtime via `len(tokenizer)`).

**Critical alignment requirements** (checked by `MANTISConfig.validate()`, which runs at construction and after each preset):
- `MetaControllerConfig.d_model` must match `BaseMoEConfig.d_model`
- `MetaControllerConfig.n_experts` must match `BaseMoEConfig.n_experts`
- `SemanticMemoryConfig.dimension` must match `BaseMoEConfig.d_model`
- `EpisodicMemoryConfig.d_model` must match `BaseMoEConfig.d_model`
- `d_model / n_heads` must be an even integer (RoPE)

Add new model sizes through `_sized_config()`, which aligns these fields. Mismatches raise `ValueError`.

### Training Pipeline

MANTIS uses a staged training pipeline:

1. **Stage 1 (REQUIRED)**: Base MoE pre-training
   - Standard next-token prediction
   - Load balancing loss for expert utilization
   - Output: Functional LLM ready for text generation

2. **Stage 2 (OPTIONAL)**: Memory fine-tuning
   - Trains episodic + semantic memory systems
   - Enables context beyond 8K tokens
   - Requires Stage 1 checkpoint

3. **Stage 4 (OPTIONAL)**: Critic training
   - Supervised correctness classifier; enables the verification gate
   - Requires Stage 1 checkpoint (config and tokenizer)

4. **Stage 3 (OPTIONAL)**: RL training
   - Trains meta-controller routing policy and state encoder with PPO
   - Optimizes accuracy/latency/compute trade-offs
   - Requires Stage 1 checkpoint; Stage 2 and Stage 4 outputs enable their gates
   - Saves `meta_controller_rl.pt`, which records component paths; `MANTISInferenceEngine.from_checkpoints(policy_checkpoint=...)` rebuilds the full engine

Stages 2 and 4 depend only on Stage 1. Training is implemented in `mantis/training/` with separate modules per stage.

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
- DeepSpeed ZeRO-2 (optional; checkpoints then omit the sharded optimizer state)
- DDP with `find_unused_parameters=True` (experts without tokens get no gradient)

**Critical**: When modifying training loop:
- Restore model state using `accelerator.unwrap_model()` before loading checkpoints
- Round `steps_per_epoch` to nearest multiple of `gradient_accumulation_steps` to prevent stale gradient leakage
- Step the LR scheduler manually once per real optimizer update; `accelerator.prepare(scheduler)` steps it once per process
- Validate tokenizer exists when resuming (`--tokenizer-path` required); checkpoints store a tokenizer fingerprint that must match
- Save checkpoints with `save_training_checkpoint()`: config, optimizer/scheduler/scaler/RNG state, and the position (`epoch` completed, `epoch_step` batches into the next). Mid-epoch checkpoints resume at the exact batch

### Dataset Handling

All modes follow `mantis/data.py`: documents are separated by blank lines (one HuggingFace example = one document), each gets one EOS, and documents are packed into windows of `seq_len + 1` tokens every `--stride` tokens.

Three dataset modes supported:

1. **Raw text** (`TextDataset`): Tokenizes on-the-fly, slower but simple
2. **Pre-tokenized** (`PreTokenizedDataset`): 5-10x faster, requires preprocessing with `scripts/preprocess_data.py`
3. **HuggingFace streaming** (`StreamingTextDataset`): No download needed, use with `--streaming` and `--steps-per-epoch`; reshuffled per epoch; cap validation with `--val-max-batches`

**WARNING**: `TextDataset` loads all tokens into RAM (2 bytes per token). For files >10GB, use `--pretokenized` or `--hf-dataset --streaming`.

### Validation Split Modes

Two validation modes:

1. **Auto-split** (`--val-split 0.1`): Convenient for iteration. Raw text splits at a document boundary (last documents), HuggingFace data splits examples, and pre-tokenized data takes the tail windows and drops the windows that would overlap it
2. **Pre-split** (`--val-file data/val.txt`): Reproducible, recommended for production

Cannot combine both. HuggingFace datasets use `--hf-val-split` instead.

### Inference Engine

Full MANTIS inference needs the base model and meta-controller; episodic memory, semantic memory and the critic are optional, and each enables its gate. `MANTISInferenceEngine.from_checkpoints()` builds the engine from saved artifacts; routing thresholds come from `InferenceConfig`. `generate()` returns only the completion.

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
- Cache shape: `(batch, n_heads, seq_len, d_head)`, keys stored after RoPE
- Cache is optional (training doesn't use it)
- Cache grows incrementally during generation; `generation.py` re-encodes the latest half-window when it would exceed `max_seq_len`

### Known Issues

1. **Semantic Memory Scaling**: RAM-based storage limits to ~100K entries on 16GB systems. Disk-backed storage planned.

2. **TextDataset Memory**: Not true streaming, loads all tokens into RAM. Use `--pretokenized` or `--streaming` for large datasets.

3. **RTX 3060 cuBLAS Bug**: Ampere GPUs have kernel bug with large vocab matrices at seq_len ≥5. Automatic workaround applied in `train.py`, `train_evo.py`, `inference.py` and `inference_evo.py`.

4. **mamba-ssm Dependency**: Requires CUDA-capable GPU for compilation and runtime. CPU-only systems cannot use episodic memory. Package imports are lazy, so Stage 1, the tokenizer and basic inference need neither mamba-ssm nor faiss (`pip install -e .[memory]` adds them).

## Important Notes

- **No trained weights**: This is a research prototype with architecture only. Full training requires 500K-1M A100 GPU-hours.
- **Validation required**: Design claims (reduced hallucinations, extended context) are unvalidated and require large-scale training + evaluation.
- **True attention limited to 8K**: Not 1M despite claims. Memory systems extend context, but base attention is 8K max.
- Always use `--tokenizer-path` when resuming training to ensure vocabulary consistency.
- When using `--resume`, the model config is loaded from checkpoint, not CLI args (except vocab_size which syncs with tokenizer).
- Gradient accumulation steps should divide evenly into steps_per_epoch to prevent stale gradients leaking across epochs.
