# HMST: Hierarchical Memory-State Transformer

A novel LLM architecture exploring hallucination mitigation and long-context memory through hierarchical memory systems and integrated self-verification.

**Includes**: Architecture implementation • Training scripts • Inference engine • Tests
**Missing**: Trained weights • Evaluation framework • Benchmark results

## Architecture

![HMST Architecture Diagram](hmst_architecture.png)

**Components**:
- **Three-tier memory**: Attention (8K) • Episodic SSM • Semantic FAISS
- **Meta-controller**: RL-trainable routing with 5 decision gates
- **MoE base model**: ~12B total, ~2B active parameters
- **Critic model**: Integrated hallucination detection

**Design Goals** (unvalidated): Reduced errors via verification • Extended context via hierarchical memory • Efficiency via sparse experts • Lower latency via early-exit

## Installation

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest tests/ -v  # Verify
python demo.py              # Run demo (untrained model)
```

## Quick Start

```bash
# Demo (untrained, shows data flow only)
python demo.py

# Training with HuggingFace datasets (no download, streaming)
python train.py --hf-dataset roneneldan/TinyStories --hf-val-split validation --streaming --steps-per-epoch 1000

# Training with local files (auto-split validation)
python train.py data/train.txt --val-split 0.1

# Production training (pre-split validation)
python train.py data/train.txt --val-file data/val.txt --output-dir checkpoints/prod

# Inference (after training)
python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time"

# Quantized inference (2-4x faster)
python inference.py checkpoints/train/best_model.pt --prompt "Once upon a time" --quantize int8
```

## Training

### HuggingFace Datasets (Easiest for Public Datasets)

```bash
# Stream without downloading (no storage needed)
python train.py --hf-dataset roneneldan/TinyStories --hf-val-split validation --streaming --steps-per-epoch 1000

# Use only 10% of dataset
python train.py --hf-dataset wikitext --hf-config wikitext-2-raw-v1 --hf-train-split "train[:10%]" --hf-val-split validation

# Use first 1000 examples (testing)
python train.py --hf-dataset openwebtext --hf-train-split "train[:1000]" --hf-val-split "train[1000:1200]"
```

### Pre-tokenization (Fastest for Local Data)

```bash
# Preprocess once, train many times
python scripts/preprocess_data.py --input data/train.txt --output data/tokenized/train
python scripts/split_dataset.py  # creates train_split and val directories
python train.py data/tokenized/train_split --pretokenized --val-file data/tokenized/val
```

### Validation Options
- `--val-split 0.1`: Auto-split for quick iteration (reshuffles each run)
- `--val-file data/val.txt`: Pre-split for reproducible results (production)
- `--hf-val-split validation`: Use HuggingFace dataset split

### Tokenizer Management
First run creates tokenizer at `<output-dir>/tokenizer`. Reuse with `--tokenizer-path` for consistency.

### Checkpoint Resumption
Resume from any checkpoint (`best_model.pt`, `epoch_N.pt`, `final_model.pt`):
```bash
# Resume training (continues from saved state)
python train.py data/train.txt \
    --resume checkpoints/train/best_model.pt \
    --tokenizer-path checkpoints/train/tokenizer \
    --val-split 0.1

# Continue for more epochs (e.g., 20 → 50 total)
python train.py data/train.txt \
    --resume checkpoints/train/final_model.pt \
    --tokenizer-path checkpoints/train/tokenizer \
    --epochs 50 \
    --val-split 0.1
```
**Note**: `--tokenizer-path` required when resuming. Restores model, optimizer, scheduler, and training state.

### Model Sizes
`micro` (~10M) • `tiny` (~100M) • `small` (~1B) • `base` (~12B)

### Common Training Patterns

```bash
# Multi-GPU with mixed precision (auto-detected)
python train.py data/train.txt --mixed-precision --val-split 0.1

# Specific GPUs only
python train.py data/train.txt --gpu-ids 0 2 --val-split 0.1

# Mixed VRAM GPUs (e.g., 12GB + 6GB): use gradient accumulation
python train.py data/train.txt \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --mixed-precision \
    --val-split 0.1
# Effective batch: 2 × 4 × N_GPUs

# Full production config
python train.py data/train.txt \
    --val-file data/val.txt \
    --model-size small \
    --mixed-precision \
    --eval-every 500 \
    --patience 3
```

### Key Training Options
`--resume` • `--hf-dataset` • `--hf-train-split` • `--hf-val-split` • `--streaming` • `--model-size` • `--val-split` / `--val-file` • `--gpu-ids` • `--gradient-accumulation-steps` • `--mixed-precision` • `--gradient-checkpointing` • `--use-8bit-optimizer` • `--eval-every` • `--patience` • `--tokenizer-path` • `--pretokenized`

See `python train.py --help` for full options.

### RL Training (Stage 3)
After pre-training, optimize meta-controller:
```bash
python -m training.rl_train --checkpoint checkpoints/prod/final.pt --episodes 50000
```

## Testing

```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_models.py -v      # Specific suite
python -m pytest tests/ --cov=hmst            # With coverage
```

## Project Structure

```
hmst/
├── models/           # base_moe, meta_controller, critic, ssm
├── memory/           # episodic, semantic, consolidation
├── training/         # pretrain, rl_train
├── inference/        # engine
├── configs/          # model_config (presets)
├── train.py          # Training script
├── inference.py      # Inference script
└── tests/            # Unit tests
```

## Troubleshooting

**OOM**: Reduce `--batch-size`, add `--mixed-precision`, `--gradient-checkpointing`, `--use-8bit-optimizer`, or use `--gradient-accumulation-steps`
**Mixed VRAM GPUs**: Use `--gradient-accumulation-steps 4 --batch-size 2`
**Slow tokenization**: Pre-tokenize with `scripts/preprocess_data.py` (5-10x faster)
**Tokenizer mismatch**: Reuse with `--tokenizer-path` when continuing training
**Resume errors**: Must use `--tokenizer-path` and matching data source as original run
**Multi-GPU issues**: Check `nvidia-smi` and available VRAM
**FAISS errors**: Requires ~10K embeddings before index training

## Validation Requirements

**Training**: 2-5T tokens • 500K-1M A100 GPU-hours • ~$1M budget
**Evaluation**: MMLU, TruthfulQA, HumanEval, GSM8K, hallucination metrics, baselines (Llama-2, Mistral)
**RL Pipeline**: Pre-trained base model • 50K episodes • Reward model
**Production**: Quantization • Serving infrastructure • Safety testing

## Component Details

**BaseMoEModel**: 8 experts • Top-2 routing • Load balancing • Scales from 100M to 12B params
**MetaController**: 6-layer transformer • 5 routing gates (early-exit, episodic/semantic memory, expert selection, verification)
**Memory**: Episodic (SSM, 8K tokens) • Semantic (FAISS RAG, 1M+ entries)
**Critic**: Hallucination detection via consistency checking

## Limitations

Untrained weights • No evaluation code • Placeholder memory consolidation • Stage 2 unimplemented • True attention: 8K (not 1M)

## Contributing & Future Work

**Contributions welcome**: Evaluation framework • Training runs • Benchmark integration • Optimizations • Bug fixes

**Roadmap**: Evaluation framework • Memory consolidation • Stage 2 training • Expert routing optimization • SSM parallelization • Custom tokenizer • Validation experiments • Full-scale training

**vs. Production LLMs** (LLaMA, Mistral, etc.): This project provides architecture + scripts but lacks trained weights, datasets, evaluation, and production optimization.

## License & Acknowledgments

MIT License. Developed with assistance from Claude (Anthropic) and Gemini (Google DeepMind).

---

**Disclaimer**: Research prototype. Performance claims are unvalidated design goals requiring large-scale training for assessment.
