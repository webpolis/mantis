# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MANTIS (Metacognitive Adaptive Network with Tiered Inference Strategies) is a research prototype LLM architecture exploring hallucination mitigation and long-context memory through metacognitive routing and hierarchical memory systems. The architecture consists of:

- **Base MoE Model**: Sparse Mixture-of-Experts transformer with grouped-query attention (~6.7B total, ~1.9B active parameters)
- **Three-tier memory**: Attention (8K) → Episodic SSM retrieval keys → Semantic FAISS with namespaces and trust levels
- **Meta-controller**: RL-trainable routing with 5 decision gates (bypass, episodic, semantic, expert bias, verification)
- **Critic model**: Verification head over the frozen backbone's hidden states, with one evidence-recovery round before abstention

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

### Stages 2-5 (OPTIONAL)

Stages 2-4 take the Stage 1 model and an optional JSONL file (demo data without it); Stage 1-only flags are rejected. Stage 5 fine-tunes the base model itself and accepts the Stage 1 flags.

```bash
# Stage 2: memory fine-tuning. JSONL: {"query", "context"}
python train.py --stage 2 data/memory.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --epochs 5 --output-dir checkpoints/stage2

# Stage 4: critic training. JSONL: {"query", "response", "evidence"? (str or list), "label"}
# Splits train/calibration/validation; fits the score temperature; reports Brier and ECE
python train.py --stage 4 data/critic.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --output-dir checkpoints/critic

# Stage 3: RL routing policy. JSONL: {"query", "answer"}. Each optional component enables its gate.
# --rl-supervised-episodes runs a route-search warm start on frozen memory before PPO.
python train.py --stage 3 data/qa.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --memory-checkpoint checkpoints/stage2/memory_system_final.pt \
    --semantic-store checkpoints/stage2/semantic_memory \
    --critic-checkpoint checkpoints/critic/critic_best.pt \
    --rl-supervised-episodes 2000 --rl-episodes 50000 --output-dir checkpoints/stage3

# Stage 5: generator adaptation (SFT in the engine's chat prompt format). JSONL: {"query", "response", "evidence"?}
# --resume gives weights only; the checkpoint records prompt_format='chat'
python train.py --stage 5 data/sft.jsonl \
    --resume checkpoints/stage1/best_model.pt \
    --tokenizer-path checkpoints/stage1/tokenizer \
    --val-split 0.1 --epochs 3 --mixed-precision bf16 --output-dir checkpoints/stage5
```

`--mixed-precision` alone means FP16; `--mixed-precision bf16` selects BF16 (train.py and train_evo.py).

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

All decode loops share `mantis/inference/generation.py`: KV cache, banned tokens (pad, bos, unk, reserved), and a context window of `max_seq_len`. When the cache fills, the latest half-window is re-encoded from scratch. `generate_tokens(prefill=(past_key_values, last_logits))` continues from an existing prefill, and `hidden_out=[...]` collects the final hidden state of every processed token; the engine uses both so a query is encoded once and its memory write needs no extra pass.

### Full Engine

```python
from mantis.inference.engine import MANTISInferenceEngine

engine = MANTISInferenceEngine.from_checkpoints(
    policy_checkpoint="checkpoints/stage3/meta_controller_rl.pt",  # records base/memory/store/critic paths
    memory_dir="runtime/memory", dtype="bfloat16")
engine.ingest(text, namespace="alice", source="user")          # chunked to the window
result = engine.generate("...", namespace="alice")             # evidence=[...] gives oracle evidence
with engine.frozen_memory():                                   # evaluation: no writes, no consolidation
    ...
engine.close()                                                 # flush consolidation, save memory_dir
```

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

# Full per-tick routing, retrieval and verification (requires compatible
# Stage 3 policy, Stage 2 memory/store and Stage 4 critic artifacts)
python inference_evo.py checkpoints/evo_train/best_model.pt \
    --new-world --policy-checkpoint checkpoints/evo_policy/meta_controller_rl.pt \
    --memory-checkpoint checkpoints/evo_memory/memory_system_final.pt \
    --semantic-store checkpoints/evo_memory/semantic_memory \
    --critic-checkpoint checkpoints/evo_critic/critic_best.pt \
    --memory-dir runtime/evo --namespace world-42
```

```python
# Python API (for web app integration)
from inference_evo import EvoInferenceEngine

engine = EvoInferenceEngine("checkpoints/evo_train/best_model.pt")
for tick in engine.generate_world(seed=42, temperature=0.7):
    send_to_client(tick)
```

With `policy_checkpoint=...` and compatible auxiliary checkpoints,
`EvoInferenceEngine` uses `MANTISInferenceEngine.generate()` per tick. It
passes the `---` token as a stop ID, uses the `trace` evidence format, and
exposes the last route/critic result in `engine.last_result`. Call
`engine.close()` to persist memory. On critic abstention, it stops rather than
emitting a non-protocol refusal. The browser playground's model streaming
path still uses the standalone backbone.

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

```bash
# Ablation controls and the synthetic multi-session memory benchmark
python scripts/run_eval.py checkpoints/stage1/best_model.pt --benchmarks memory \
    --policy-checkpoint checkpoints/stage3/meta_controller_rl.pt \
    --route-policy always --expert-bias --dtype bfloat16 --memory-mode frozen --records records.jsonl
python scripts/run_eval.py checkpoints/stage1/best_model.pt --benchmarks memory --memory-bench-mode prompt
```

The engine's confidence is the critic score when verified, the geometric-mean token probability otherwise, and 0 after an abstention (`confidence_source` says which). Metrics: accuracy, error rate, coverage, answered error rate, confident-error rate (wrong with confidence ≥ 0.8; not a hallucination rate), ECE, Brier, AURC. TruthfulQA is a lexical proxy (`truthfulness_proxy`), MMLU reads the first standalone letter of at most 10 new tokens, HumanEval runs code in Docker without network when available (else a resource-limited subprocess that is not a security boundary). Benchmarks run under `frozen_memory()` by default; the memory benchmark (`evaluation/memory_bench.py`) is stateful within its own namespace and has a `prompt` mode that prepends all facts as a recent-text-buffer baseline. Reports carry p50/p95 latency, mean `compute_units`, peak GPU memory, artifact versions and optional per-example records.

Tests: `python -m pytest tests -q` (deterministic diagnostics; episodic/semantic tests skip without mamba-ssm/faiss).

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
│   ├── episodic.py      # SSM-keyed recent interactions with provenance and hit counts
│   ├── semantic.py      # FAISS-based long-term memory (namespaces, trust, tombstones, fingerprint)
│   └── consolidation.py # Lifecycle: overflow queue, periodic promotion, persistence
├── training/            # Training pipelines
│   ├── common.py        # Accelerator, optimizer, LR schedule shared by train.py/train_evo.py
│   ├── pretrain.py      # Standalone single-GPU Stage 1 trainer (train.py is the main path)
│   ├── memory_train.py  # Stage 2: Memory fine-tuning
│   ├── rl_train.py      # Stage 3: RL meta-controller training (+ supervised route search)
│   ├── critic_train.py  # Stage 4: Critic training with calibration split
│   ├── sft.py           # Stage 5: instruction / evidence dataset in the engine's prompt format
│   └── scoring.py       # Lexical answer correctness shared by rewards and evaluation
├── inference/           # Generation engine
│   ├── generation.py    # Shared decode loop (cache, window, sampling, prefill reuse)
│   ├── prompting.py     # Query formats, evidence block with source ids, budgeted selection
│   └── engine.py        # Dynamic routing, retrieval, verification, memory writes
├── simulation/          # Ecological simulator for evolution training data (see its CLAUDE.md)
├── configs/             # Configuration management
│   └── model_config.py  # Presets: micro/tiny/small/base
├── utils/checkpoints.py # Checkpoint schema, model/tokenizer loading
├── data.py              # Documents, EOS, packing, leak-free splits
└── tokenizer.py         # MANTISTokenizer (trie-based, 512 tokens, byte fallback)

evaluation/              # Evaluation harness
├── benchmarks.py        # MMLU, TruthfulQA (proxy), HumanEval (docker sandbox), GSM8K
├── memory_bench.py      # Synthetic multi-session memory benchmark (memory vs prompt baseline)
└── metrics.py           # Accuracy, error/coverage, confident-error rate, ECE, Brier, AURC
tests/                   # Deterministic diagnostics (pytest)

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
- Load balancing loss (top-1 assignments), weighted by `load_balance_weight`; the output's `expert_load` (n_moe_layers, n_experts) reports top-k dispatch fractions, the work actually sent to each expert
- Grouped-query attention: `n_kv_heads` KV heads shared by `n_heads` query heads; the cache is `(batch, n_kv_heads, seq, d_head)`
- Scales from 3M (micro, dense) to 6.7B parameters (base)
- Pre-norm transformer backbone with rotary positional embeddings and `scaled_dot_product_attention`
- `max_seq_len` is the attention window; new models set it to the training `--seq-len`
- `return_hidden=True` returns only the final normalized hidden states (`last_hidden`)
- The per-expert Python loop is reference code; profile before replacing it with grouped GEMMs

**Important**: The model uses `top_k=2` sparse routing by default. When modifying expert selection, ensure load balancing loss is maintained to prevent expert collapse.

**Meta-controller integration**: `expert_weights` is `(batch, n_layers, n_experts)`, one bounded bias vector per layer, added to that layer's gate logits. The controller produces `expert_bias_scale * tanh(raw)` from a zero-initialized head, and the engine applies it only when `InferenceConfig.expert_bias` is on (off by default). Whether learned routing survives depends on the bias magnitude, so treat it as an ablation.

#### MetaController (`mantis/models/meta_controller.py`)

RL-trainable routing controller with 5 gates:
1. **Bypass**: Skip the optional components (memory reads, expert bias, verification) and decode from the query's cached prefill. The backbone still runs at full depth
2. **Episodic memory**: Access recent context (8K tokens)
3. **Semantic memory**: Retrieve long-term facts (1M+ entries)
4. **Expert bias**: Bounded per-layer additive bias to MoE routing, off by default
5. **Verification**: Trigger critic model

A residual MLP over the pooled query embedding and a state summary (query next-token entropy and top-1 probability, context fill, memory fill; these measure query predictability, not answer correctness). Trained with PPO in Stage 3 as a stochastic policy: gates 1-3 and 5 are Bernoulli, and gate 4 is a Gaussian over the raw bias before `tanh`. Deterministic inference thresholds gate probabilities (`InferenceConfig.routing_threshold`) and uses the Gaussian mean. `log_prob` takes a 5-entry action mask: gates whose component is missing, gates the bypass path ignored, and an unused expert bias add nothing. `InferenceConfig.route_policy` (`learned | always | never | bypass`) and `generate(force_gates=...)` replace the policy for ablations and route search.

#### Memory Systems

**Episodic Memory** (`mantis/memory/episodic.py`):
- Mamba SSM (mamba-ssm library, CUDA-optimized) encodes each interaction independently into a 256-d retrieval key; it is learned retrieval, not a recurrent state carried across turns
- Entries hold role-labelled token segments (`query`/`response` or `document`), the key, a pooled `d_model` embedding for reranking, a hit count and metadata (`namespace`, `source`, `trust`, `verified`, `timestamp`); full hidden states are not kept
- `add(hidden, segments, metadata)` takes the hidden states the engine already computed during decoding; at capacity the oldest entry goes to `on_overflow` (the consolidator) before it is dropped
- `retrieve(query_hidden, top_k, namespace, exclude_ids)` returns entry dicts with a `score` and increments hits; `candidates(min_hits)` feeds consolidation; `save`/`load` persist the buffer
- The engine stores every answered interaction with provenance and never stores an abstention; input beyond the window is ingested as document chunks (`engine.ingest`)

**Semantic Memory** (`mantis/memory/semantic.py`):
- FAISS vector database with stable IDs; eviction and `delete()` tombstone IDs, searches over-fetch by a bounded amount, and the index rebuilds off-thread with an atomic swap when >20% are stale
- IVF serves exact search until 10K entries, then trains IVF-PQ with a cluster count sized to the data; retrieval is approximate from then on and `recall_at_k()` measures it against exact search
- Metadata: `namespace` (per caller; Stage 2 facts live in `global`), `source` → `trust` (user/external/stage2 = 2, verified model = 1, unverified = 0), `superseded_by` (via `supersede()`; superseded entries stay as records but are not retrieved). `retrieve_with_metadata(namespaces=, min_trust=, exclude_ids=)` filters; the engine's `min_evidence_trust` (default 1) keeps unverified generated claims out of the facts prompt
- `embedding_fingerprint` (tokenizer + full-backbone hash from `model_fingerprint()`) is saved with the store and checked by `from_checkpoints`; new Stage 2, 3, and 4 checkpoints are checked against the same backbone identity
- Optional projection (trained in Stage 2) applied before L2-normalized indexing
- **WARNING**: Keeps full FP32 vectors in RAM (8.2GB for 1M entries at d_model 2048, before text and index). Limit to ~100K entries on 16GB RAM systems.

**Consolidation** (`mantis/memory/consolidation.py`):
- `from_checkpoints` builds and starts it when both memories exist; `engine.close()` stops it, flushes the queue and saves `memory_dir`
- Every evicted episodic entry is written to semantic memory as its own record (text, stored pooled embedding, provenance, `origin: episodic:<id>`) through a bounded queue; a full queue writes synchronously. Nothing is summarized away
- The periodic cycle (`consolidation_interval`) promotes entries with `hits >= consolidation_min_hits` or `metadata['important']`, and removes from the buffer only those whose write succeeded
- `pause()`/`resume()` back `engine.frozen_memory()`; `get_stats()` reports stored, duplicates, failures, queue lag

#### Critic Model (`mantis/models/critic.py`)

Small encoder (6 layers; ~80M parameters at `base`) over the frozen base model's final hidden states of `[evidence; query; response]` (evidence first so the causal backbone lets response tokens see it). `build_input` gives the response 50%, the evidence 35% and the query 15% of `max_seq_len`, then redistributes leftover capacity in that order. One correctness head; `temperature` (a buffer fitted in Stage 4 on a calibration split) scales the logit. `verify(base_model, query_ids, response_ids, evidence_ids)` returns the calibrated probability. The engine passes the same evidence lines the generator saw, retries retrieval once with the draft when the score is below `verification_confidence_threshold` (`verification_retries`), and abstains if it still fails.

### Model Configuration

Model sizes are defined in `mantis/configs/model_config.py`:

- **micro**: ~3M parameters (dense, not MoE), controller 0.6M, critic 2.2M - ultra-fast testing
- **tiny**: ~55M parameters, ~30M active (4 experts, 8/4 heads), controller 2.4M, critic 14M - development/debugging
- **small**: ~435M parameters, ~234M active (4 experts, 32/8 heads), controller 18M, critic 45M - experimentation
- **base**: ~6.7B parameters, ~1.9B active (8 experts, 32/8 heads), controller 106M, critic 80M - production target

`get_large_config()` (~71B, 16 experts) and `get_extmem_config()` (32K windows) exist too, but `--model-size` does not offer them.

**Vocabulary**: All models use 512 tokens (custom domain-specific trie tokenizer, synced at runtime via `len(tokenizer)`).

**Critical alignment requirements** (checked by `MANTISConfig.validate()`, which runs at construction and after each preset):
- `MetaControllerConfig.d_model` must match `BaseMoEConfig.d_model`
- `MetaControllerConfig.n_experts` must match `BaseMoEConfig.n_experts`
- `SemanticMemoryConfig.dimension` must match `BaseMoEConfig.d_model`
- `EpisodicMemoryConfig.d_model` must match `BaseMoEConfig.d_model`
- `d_model / n_heads` must be an even integer (RoPE); `n_kv_heads` must divide `n_heads`
- `CriticConfig.max_seq_len` cannot exceed `BaseMoEConfig.max_seq_len` (the critic reads base hidden states)
- `InferenceConfig.route_policy` and `prompt_format` must be known values

Add new model sizes through `_sized_config()`, which aligns these fields and sizes the controller and critic. Mismatches raise `ValueError`. `InferenceConfig` holds the engine knobs: `bypass_uncertainty_threshold`, `verification_confidence_threshold`, `verification_retries`, `expert_bias`, `route_policy`, `prompt_format`, `episodic_top_k`/`semantic_top_k`, `min_evidence_trust`, `remember`. `TrainingConfig` holds the Stage 3 reward weights plus `latency_budget_s`, `abstain_reward`, `correctness_f1_threshold`, and `sft_lr` for Stage 5.

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
   - Trains meta-controller routing policy and state encoder with PPO; optional supervised route-search warm start on frozen memory
   - Reward: lexical correctness (`mantis/training/scoring.py`; abstentions get `abstain_reward`), latency over `latency_budget_s`, measured `compute_units` beyond a query-only answer, calibration of the final confidence
   - Training episodes write to memory (sequential, stateful); validation runs under `frozen_memory()` and reports accuracy, coverage, answered error rate, p95 latency, mean compute
   - Requires Stage 1 checkpoint; Stage 2 and Stage 4 outputs enable their gates
   - Saves `meta_controller_rl.pt`, which records component paths; `MANTISInferenceEngine.from_checkpoints(policy_checkpoint=...)` rebuilds the full engine

5. **Stage 5 (OPTIONAL)**: Generator adaptation
   - Fine-tunes the base model on {"query", "response", "evidence"?} JSONL in the engine's chat format (`mantis/training/sft.py`, built with `mantis/inference/prompting.py`); loss on response tokens only
   - Runs through the Stage 1 `train()` loop; `--resume` supplies weights only; the checkpoint records `prompt_format='chat'`

Stages 2, 4 and 5 depend only on Stage 1. Training is implemented in `mantis/training/` with separate modules per stage.

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

Full MANTIS inference needs the base model and meta-controller; episodic memory, semantic memory and the critic are optional, and each enables its gate. `MANTISInferenceEngine.from_checkpoints(..., dtype=, memory_dir=)` builds the engine from saved artifacts, checks the semantic store's embedding fingerprint, and starts the consolidator when both memories exist; call `close()` to flush it. `generate()` returns only the completion plus metadata: `path` (`bypass`/`full`), `confidence` and `confidence_source` (`critic`/`token_likelihood`/`abstained`), `critic_score`, `abstained`, `verification_rounds`, `evidence` (source ids and scores), `timings` per component, `cost` (token counts) and `compute_units` (token × parameter products over backbone, critic and SSM).

For basic generation (Stage 1 only), use `inference.py` which provides simplified inference without memory systems.

Production inference engine is in `mantis/inference/engine.py` and coordinates:
1. Query encoding, once (cache and hidden states are reused); input beyond the window is ingested into memory first
2. Query predictability features and meta-controller routing (or a fixed `route_policy`)
3. Bypass: decode from the cached prefill, nothing else
4. Full path: retrieve candidates from the open tiers in the caller's namespace (+ `global`), rerank (tier score + word F1), fit them into the evidence budget with a share per tier, prepend them with source ids (`mantis/inference/prompting.py`); reuse the prefill when there is no evidence and no expert bias
5. Optional critic verification with the same evidence; on rejection, one more retrieval round conditioned on query + draft, then abstention
6. Episodic write of the answered interaction with provenance, from the hidden states decoding already produced (never abstentions; disabled under `frozen_memory()`)

### KV Caching

KV caching is implemented in `BaseMoEModel` for efficient inference. When adding new attention mechanisms, ensure:
- Cache shape: `(batch, n_kv_heads, seq_len, d_head)`, keys stored after RoPE (the `base` preset's 8K cache is 0.375 GiB in 16-bit precision)
- Cache is optional (training doesn't use it)
- Cache grows incrementally during generation; `generation.py` re-encodes the latest half-window when it would exceed `max_seq_len`
- A prefill can only be reused when the prompt tokens and `expert_weights` are unchanged; prepending evidence or changing the bias alters every hidden state

### Known Issues

1. **Semantic Memory Scaling**: RAM-based storage (full FP32 vectors plus the index) limits to ~100K entries on 16GB systems. Disk-backed storage planned. Each IVF rebuild retrains the codebooks over all live vectors.

2. **TextDataset Memory**: Not true streaming, loads all tokens into RAM. Use `--pretokenized` or `--streaming` for large datasets.

3. **RTX 3060 cuBLAS Bug**: Ampere GPUs have kernel bug with large vocab matrices at seq_len ≥5. Automatic workaround applied in `train.py`, `train_evo.py`, `inference.py` and `inference_evo.py`.

4. **mamba-ssm Dependency**: Requires CUDA-capable GPU for compilation and runtime. CPU-only systems cannot use episodic memory. Package imports are lazy, so Stage 1, the tokenizer and basic inference need neither mamba-ssm nor faiss (`pip install -e .[memory]` adds them).

5. **Lexical Scoring**: The Stage 3 reward and the TruthfulQA runner compare text lexically (`scoring.answer_correct`: normalized phrase match without negation words, or word F1 ≥ 0.5). They reject negated answers but misjudge paraphrases and verbose correct answers.

6. **Tokenizer**: The 512-token vocabulary makes natural-language sequences several times longer than a subword vocabulary would, so per-token costs and context coverage are not comparable with subword models. A general-language tokenizer is future work.

## Important Notes

- **No trained weights**: This is a research prototype with architecture only. Full training of the `base` preset on 2-5T tokens needs roughly 50K-125K A100 GPU-hours by a parameter-dominated estimate (6 × 1.9B active × tokens at 125 TFLOPS sustained), before attention overhead, auxiliary training, evaluation and failed runs.
- **Validation required**: Design claims (reduced hallucinations, extended context, lower cost) are unvalidated. The review's gates: beat ordinary retrieval on the memory benchmark, lower measured cost at comparable quality, lower answered-error rate at matched coverage; run the ablation ladder (`run_eval.py --route-policy`, `--expert-bias`, `--memory-bench-mode prompt`) before scaling.
- **True attention limited to the preset window (8K)**: Memory systems extend what the generator can see only as far as retrieval recall and the evidence budget allow; a 1M-entry store is datastore capacity, not context.
- **Bypass is not depth reduction**: Gate 1 skips memory reads, expert bias and verification; the backbone always runs at full depth.
- Always use `--tokenizer-path` when resuming training to ensure vocabulary consistency.
- When using `--resume`, the model config is loaded from checkpoint, not CLI args (except vocab_size which syncs with tokenizer).
- Gradient accumulation steps should divide evenly into steps_per_epoch to prevent stale gradients leaking across epochs.
