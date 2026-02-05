# HMST: Hierarchical Memory-State Transformer

> **⚠️ Research Prototype**: This repository contains a proof-of-concept implementation of a novel LLM architecture. The system has **not been trained to completion** and performance claims are **theoretical design goals**, not validated results. This is an architectural exploration and research prototype, not a production-ready model.

A novel LLM architecture design exploring hallucination mitigation and long-context memory through intelligent memory management and integrated self-verification.

## Project Status

**What This Repository Contains:**
- ✅ Complete implementation of the proposed architecture components
- ✅ Training scripts for pre-training and RL optimization
- ✅ Inference engine with dynamic routing
- ✅ Unit tests for core functionality

**What This Repository Does NOT Contain:**
- ❌ Trained model weights or checkpoints
- ❌ Evaluation framework or benchmark results
- ❌ Validated performance metrics
- ❌ Production-ready tokenizers (uses placeholder implementation)

**Current State**: The architecture is implemented but requires large-scale training (estimated $750K-$1.5M in compute costs for 12B parameter variant) to validate its design hypotheses.

## Architecture Overview

HMST proposes a novel architecture combining:
- **Three-tier memory system**: Working memory (Attention), Episodic memory (SSM), Semantic memory (FAISS)
- **RL-trainable meta-controller**: Dynamic routing with 5 decision gates
- **Mixture-of-Experts base model**: Designed for ~12B total, ~2B active parameters
- **Integrated critic model**: Hallucination detection and verification component

### Design Goals (Unvalidated)

The architecture is designed to achieve:
- Reduced factual errors through critic-based verification
- Extended context handling via hierarchical memory (attention: 8K tokens + RAG-style retrieval)
- Improved compute efficiency through sparse expert activation
- Lower latency via early-exit mechanisms

**Note**: These are architectural design targets, not measured results. Validation would require full-scale training and comprehensive evaluation.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hmst.git
cd hmst

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -m pytest tests/ -v

# Run demo (uses untrained model with random weights)
python demo.py
```

## Quick Start

### Running the Demo

```bash
# Run interactive demo (untrained model)
python demo.py
```

**Note**: The demo uses an untrained model and will not produce meaningful outputs. It demonstrates the architecture's data flow and routing decisions.

### Inference

After training, use `inference.py` for text generation:

```bash
# Interactive mode (default)
python inference.py checkpoints/train/best_model.pt

# Single prompt
python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time"

# Batch generation
python inference.py checkpoints/train/best_model.pt --input prompts.txt --output results.txt

# Greedy decoding (deterministic)
python inference.py checkpoints/train/best_model.pt --prompt "Hello" --temperature 0
```

**Options**: `--max-length`, `--temperature`, `--top-p`, `--top-k`, `--output`

For full HMST system with memory/routing (research), use `hmst.inference.HMSTInferenceEngine` (requires all components trained).

## Training Quickstart

```bash
# Basic training
python train.py data/train.txt

# Production training
python train.py data/train.txt \
    --val-file data/val.txt \
    --output-dir checkpoints/prod \
    --model-size small \
    --multi-gpu \
    --mixed-precision \
    --patience 3
```

**Pre-tokenization (recommended, 5-10x faster)**:
```bash
python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
python train.py data/tokenized/train --pretokenized
```

**Note**: Base 12B model requires ~500K-1M A100 GPU-hours ($750K-$1.5M) for competitive performance.

## Training

### Basic Training

The unified training script supports single-GPU and multi-GPU training with comprehensive options:

```bash
# Basic training
python train.py data/train.txt

# With validation and custom output directory
python train.py data/train.txt \
    --val-file data/val.txt \
    --output-dir checkpoints/my_run

# View all options
python train.py --help
```

### Tokenizer Management

The script automatically manages tokenization:

```bash
# First run - creates new tokenizer (saved to output-dir/tokenizer)
python train.py data/train.txt --output-dir checkpoints/run1

# Subsequent runs - reuse existing tokenizer for consistency
python train.py data/new_train.txt \
    --tokenizer-path checkpoints/run1/tokenizer \
    --output-dir checkpoints/run2
```

**Important**: Always use the same tokenizer when continuing training or fine-tuning a model to ensure vocabulary consistency.

### Model Sizes

Choose from three preset configurations:

```bash
# Tiny (~100M params) - fast training, testing
python train.py data/train.txt --model-size tiny

# Small (~1B params) - experimentation
python train.py data/train.txt --model-size small

# Base (~12B params) - full-scale training
python train.py data/train.txt --model-size base --multi-gpu
```

### Advanced Training Options

```bash
# Control steps per epoch (useful for large datasets)
python train.py data/train.txt --steps-per-epoch 1000 --epochs 50

# Multi-GPU training with DDP
python train.py data/train.txt --multi-gpu --batch-size 4

# Mixed precision for faster training
python train.py data/train.txt --mixed-precision

# Early stopping and validation
python train.py data/train.txt \
    --val-file data/val.txt \
    --eval-every 500 \
    --patience 5

# Full production training
python train.py data/train.txt \
    --val-file data/val.txt \
    --output-dir checkpoints/production \
    --tokenizer-path checkpoints/run1/tokenizer \
    --model-size small \
    --epochs 20 \
    --batch-size 8 \
    --learning-rate 3e-4 \
    --steps-per-epoch 2000 \
    --eval-every 500 \
    --patience 3 \
    --save-every 5 \
    --multi-gpu \
    --mixed-precision
```

### Training Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--model-size` | Model size: tiny/small/base | tiny |
| `--epochs` | Number of training epochs | 20 |
| `--batch-size` | Batch size per GPU | 8 |
| `--learning-rate` | Peak learning rate | 3e-4 |
| `--steps-per-epoch` | Max steps per epoch | Full dataset |
| `--tokenizer-path` | Path to existing tokenizer | Create new |
| `--val-file` | Validation text file | None |
| `--eval-every` | Evaluate every N steps | Off |
| `--patience` | Early stopping patience (epochs) | Off |
| `--save-every` | Save checkpoint every N epochs | Off |
| `--multi-gpu` | Enable multi-GPU DDP training | False |
| `--mixed-precision` | Enable FP16 training | False |
| `--warmup-steps` | Learning rate warmup steps | 1000 |
| `--seq-len` | Sequence length | 512 |
| `--num-workers` | DataLoader workers | 4 |

### Stage 3: RL Training (Meta-Controller)

After pre-training, optimize the meta-controller with reinforcement learning:

```bash
python -m training.rl_train \
    --checkpoint checkpoints/production/final.pt \
    --episodes 50000
```

**Note**: Stage 2 (memory fine-tuning) is not yet implemented. RL training requires a pre-trained base model to generate meaningful reward signals.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_models.py -v
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_inference.py -v

# Run with coverage
python -m pytest tests/ --cov=hmst --cov-report=html
```

## Project Structure

```
hmst/
├── models/           # Core model implementations
│   ├── base_moe.py          # Sparse MoE transformer
│   ├── meta_controller.py   # RL-trainable routing controller
│   ├── critic.py            # Hallucination detection
│   └── ssm.py               # State space model (Mamba)
├── memory/           # Memory hierarchy
│   ├── episodic.py          # L2 cache (SSM-compressed)
│   ├── semantic.py          # L3 cache (FAISS vector store)
│   └── consolidation.py     # Memory transfer logic
├── training/         # Training modules
│   ├── pretrain.py          # Pre-training trainer class
│   └── rl_train.py          # RL meta-controller training
├── inference/        # Inference engine
│   └── engine.py            # Full system orchestration
├── configs/          # Model configurations
│   └── model_config.py      # Tiny/Small/Base presets
├── tokenizer/        # Tokenization
│   └── hmst_tokenizer.py    # HuggingFace BPE wrapper
├── train.py          # Production training script
├── inference.py      # Production inference script
├── demo.py           # Interactive demo
└── tests/            # Unit tests
```

## Common Issues & Tips

### Training Performance

- **Large datasets**: Use `--steps-per-epoch` to control epoch length and enable more frequent validation
- **Out of memory**: Reduce `--batch-size`, use `--mixed-precision`, or switch to smaller model size
- **Multi-GPU not working**: Ensure NCCL is installed and GPUs are visible with `nvidia-smi`
- **Slow data loading**: Increase `--num-workers` (typically 4-8 works well)

### Tokenizer Issues

- **Vocabulary mismatch**: Always use `--tokenizer-path` when continuing training or fine-tuning
- **Token IDs out of range**: Ensure the same tokenizer is used for training and inference
- **Find existing tokenizer**: Check `checkpoints/<run_name>/tokenizer` directory

### Memory Systems

- **FAISS index errors**: Semantic memory requires ~10K embeddings before index training
- **SSM dimension mismatch**: Ensure `d_model` → `d_state` projection is configured correctly
- **Memory consolidation**: Background process is placeholder; needs base model integration

## What Would Be Needed for Validation

To validate the architectural design goals, the following would be required:

### 1. Large-Scale Training
- Multi-trillion token dataset (e.g., The Pile, RedPajama, proprietary data)
- 500K-1M A100 GPU-hours for 12B parameter model
- Distributed training infrastructure (100+ GPUs)
- ~$1M budget for compute resources

### 2. Evaluation Framework
Implementation needed for:
- **Standard Benchmarks**: MMLU, TruthfulQA, HumanEval, GSM8K
- **Hallucination Metrics**: Factual consistency evaluation
- **Efficiency Metrics**: FLOPs per token, throughput measurements
- **Long-Context Tests**: "Needle in a haystack" evaluations
- **Comparison Baselines**: Run same tests on Llama-2, Mistral, etc.

### 3. RL Training Pipeline
- Train base model to convergence first
- Collect trajectory data for meta-controller optimization
- Reward model for accuracy/efficiency trade-offs
- ~50K episodes of RL training

### 4. Production Infrastructure
- Proper tokenizer training (BPE/SentencePiece)
- Model quantization and optimization
- Serving infrastructure
- Safety and alignment testing

## Architecture Details

### Component Specifications

**BaseMoEModel** (`models/base_moe.py`):
- 8 experts with top-2 routing (configurable)
- Standard transformer attention
- Load balancing loss for expert utilization
- Parameter count scales with config (tiny: ~100M, small: ~1B, base: ~12B)

**MetaController** (`models/meta_controller.py`):
- 6-layer lightweight transformer
- 5 routing decisions per query:
  - Early exit gate (skip deep processing)
  - Episodic memory access (recent context)
  - Semantic memory retrieval (long-term facts)
  - Expert selection (MoE routing weights)
  - Verification trigger (critic activation)

**Memory Systems** (`memory/`):
- **Episodic**: SSM-based compression of recent 8K token windows
- **Semantic**: FAISS vector database for retrieval (RAG-style)
- **Note**: "1M+ context" claim refers to retrieval database size, not attention window

**CriticModel** (`models/critic.py`):
- Verification model for hallucination detection
- Checks consistency between query, response, and retrieved facts
- Outputs confidence score

## Known Limitations

1. **Untrained weights**: All models use random initialization
2. **Placeholder tokenizer**: Falls back to random tokens when no tokenizer provided
3. **No evaluation code**: `evaluation/` directory is empty
4. **Memory consolidation**: Importance scoring is placeholder
5. **Stage 2 training**: Memory fine-tuning not implemented
6. **Context window**: True attention context is 8K tokens, not 1M (semantic memory uses RAG pattern)

## Comparison to Production LLM Projects

Production-ready open-source LLM projects typically include:
- ✅ Trained model weights (HuggingFace model hub)
- ✅ Training datasets and recipes
- ✅ Evaluation frameworks and benchmark results
- ✅ Tokenizers (trained BPE/SentencePiece)
- ✅ Inference optimization (quantization, serving)

This project currently provides:
- ✅ Architecture implementation
- ✅ Training scripts (untested at scale)
- ❌ Everything else listed above

Examples of complete projects: LLaMA, Mistral, Pythia, Falcon, GPT-NeoX

## Contributing

Contributions are welcome, especially:
- Evaluation framework implementation
- Training runs and results sharing
- Benchmark integration (lm-evaluation-harness)
- Optimization improvements
- Bug fixes and tests

## Future Work

- [ ] Implement evaluation framework
- [ ] Complete memory consolidation logic
- [ ] Add Stage 2 (memory fine-tuning) training
- [ ] Optimize expert routing (vectorize nested loops)
- [ ] Parallelize SSM sequential scan
- [ ] Integrate proper tokenizer training
- [ ] Conduct small-scale validation experiments
- [ ] Full-scale training (requires funding)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This architecture was developed with assistance from Claude (Anthropic) and Gemini (Google DeepMind) AI assistants during the design and implementation phases.

## Contact

For questions, collaboration, or discussions about the architecture: [Contact Information]

---

**Disclaimer**: This is a research prototype exploring novel architectural ideas for LLM development. All performance claims are theoretical design goals, not validated measurements. Large-scale training and comprehensive evaluation would be required to assess the architecture's actual capabilities.
