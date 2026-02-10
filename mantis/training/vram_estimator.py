"""
VRAM-aware batch size estimation for MANTIS training.

Analytically estimates training memory from model config + training flags,
then derives the maximum batch size that fits on each GPU.
"""


def estimate_model_params(config):
    """
    Count parameters analytically from BaseMoEConfig fields.

    Args:
        config: BaseMoEConfig (or object with matching fields)

    Returns:
        dict with 'total', 'embedding', 'per_layer_attn', 'per_layer_moe',
        'per_layer_norms', 'final_norm', 'n_layers'
    """
    V = config.vocab_size
    D = config.d_model
    L = config.n_layers
    H = config.n_heads
    F = config.d_ff
    E = config.n_experts
    S = config.max_seq_len

    # Embeddings (lm_head is tied to token_embedding, so 0 extra)
    embed_params = V * D + S * D

    # Per-layer attention: nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    #   in_proj_weight: 3 * D * D, in_proj_bias: 3 * D
    #   out_proj.weight: D * D, out_proj.bias: D
    attn_params = 4 * D * D + 4 * D

    # Per-layer norms: attn_norm + ff_norm, each LayerNorm(D) = weight(D) + bias(D)
    norm_params = 4 * D

    # Per-layer MoE:
    #   gate: Linear(D, E) = D * E + E
    #   E experts, each: w1(D, F) + w2(F, D) = 2*D*F + F + D
    gate_params = D * E + E
    expert_params = E * (2 * D * F + F + D)
    moe_params = gate_params + expert_params

    per_layer = attn_params + norm_params + moe_params

    # Final norm: LayerNorm(D) = 2 * D
    final_norm = 2 * D

    total = embed_params + L * per_layer + final_norm

    return {
        'total': total,
        'embedding': embed_params,
        'per_layer_attn': attn_params,
        'per_layer_moe': moe_params,
        'per_layer_norms': norm_params,
        'final_norm': final_norm,
        'n_layers': L,
    }


def estimate_training_vram(
    config,
    seq_len,
    batch_size,
    mixed_precision=False,
    gradient_checkpointing=False,
    use_8bit_optimizer=False,
    deepspeed_zero_stage=0,
    num_gpus=1,
):
    """
    Estimate training VRAM usage in bytes.

    Args:
        config: BaseMoEConfig
        seq_len: Training sequence length
        batch_size: Per-GPU batch size
        mixed_precision: Whether FP16 mixed precision is used
        gradient_checkpointing: Whether gradient checkpointing is enabled
        use_8bit_optimizer: Whether 8-bit AdamW is used
        deepspeed_zero_stage: 0 (none), 2, or 3
        num_gpus: Number of GPUs (for ZeRO sharding calculation)

    Returns:
        dict with 'model_weights', 'optimizer_state', 'gradients',
        'activations', 'cuda_overhead', 'total' (all in bytes),
        plus 'fixed' and 'per_sample' subtotals
    """
    params = estimate_model_params(config)
    n_params = params['total']

    D = config.d_model
    L = config.n_layers
    H = config.n_heads
    F = config.d_ff
    E = config.n_experts
    K = config.top_k
    V = config.vocab_size

    # --- Fixed costs (independent of batch_size) ---

    # Model weights: always FP32 (Accelerate autocasts on the fly)
    model_bytes = n_params * 4

    # Optimizer state: AdamW stores m + v (FP32 each)
    if use_8bit_optimizer:
        optim_bytes = n_params * 2  # 8-bit: 1 byte each for m, v
    else:
        optim_bytes = n_params * 8  # FP32: 4 bytes each for m, v

    # Gradients: FP32
    grad_bytes = n_params * 4

    # DeepSpeed ZeRO sharding
    shard = max(1, num_gpus) if deepspeed_zero_stage > 0 else 1
    if deepspeed_zero_stage >= 3:
        model_bytes //= shard
        optim_bytes //= shard
        grad_bytes //= shard
    elif deepspeed_zero_stage >= 2:
        optim_bytes //= shard
        grad_bytes //= shard

    cuda_overhead = 200 * 1024 * 1024  # ~200 MB (CUDA context + allocator bookkeeping)

    fixed = model_bytes + optim_bytes + grad_bytes + cuda_overhead

    # --- Variable costs (per sample) ---

    # Bytes per activation element
    act_elem = 2 if mixed_precision else 4

    if gradient_checkpointing:
        # Only save layer boundary activations + one full layer recompute
        # Boundary saves: L layers × seq_len × D
        boundary = L * seq_len * D * act_elem
        # One full layer recompute during backward:
        #   attention Q/K/V + scores + output + MoE intermediates
        one_layer = (
            3 * seq_len * D  # Q, K, V
            + H * seq_len * seq_len  # attention scores
            + seq_len * D  # attention output
            + K * seq_len * F  # MoE FFN intermediate
            + K * seq_len * D  # MoE FFN output
            + 4 * seq_len * D  # norms, residuals
        ) * act_elem
        activation_per_sample = boundary + one_layer
    else:
        # Full activation storage per layer + autograd tape (~2×)
        per_layer_acts = (
            3 * seq_len * D  # Q, K, V
            + H * seq_len * seq_len  # attention scores
            + seq_len * D  # attention output
            + K * seq_len * F  # MoE FFN intermediate
            + K * seq_len * D  # MoE FFN output
            + 4 * seq_len * D  # norms, residuals, masks
        )
        # Autograd tape roughly doubles activation storage
        activation_per_sample = L * per_layer_acts * 2 * act_elem

    # Input tensors: input_ids + labels (int64)
    input_per_sample = seq_len * 8 * 2

    # Final logits: seq_len × vocab_size
    logits_per_sample = seq_len * V * act_elem

    per_sample = activation_per_sample + input_per_sample + logits_per_sample

    total = fixed + per_sample * batch_size

    return {
        'model_weights': model_bytes,
        'optimizer_state': optim_bytes,
        'gradients': grad_bytes,
        'activations': per_sample * batch_size,
        'cuda_overhead': cuda_overhead,
        'total': total,
        'fixed': fixed,
        'per_sample': per_sample,
        'n_params': n_params,
    }


def compute_optimal_batch_sizes(
    config,
    seq_len,
    gpu_vram_list,
    safety_margin=0.85,
    mixed_precision=False,
    gradient_checkpointing=False,
    use_8bit_optimizer=False,
    deepspeed_zero_stage=0,
    num_gpus=1,
    max_batch_size=None,
):
    """
    Given a list of per-GPU VRAM (bytes), return the max batch_size per GPU.

    Args:
        config: BaseMoEConfig
        seq_len: Training sequence length
        gpu_vram_list: List of per-GPU total VRAM in bytes
        safety_margin: Fraction of VRAM to target (default 0.85)
        mixed_precision: Whether FP16 mixed precision is used
        gradient_checkpointing: Whether gradient checkpointing is enabled
        use_8bit_optimizer: Whether 8-bit AdamW is used
        deepspeed_zero_stage: 0, 2, or 3
        num_gpus: Number of GPUs (for ZeRO sharding)
        max_batch_size: Ceiling batch size (user's --batch-size value)

    Returns:
        list[int] of batch sizes, one per GPU
    """
    # Get fixed cost and per-sample cost (use batch_size=1 to extract)
    est = estimate_training_vram(
        config, seq_len, batch_size=1,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        use_8bit_optimizer=use_8bit_optimizer,
        deepspeed_zero_stage=deepspeed_zero_stage,
        num_gpus=num_gpus,
    )

    fixed = est['fixed']
    per_sample = est['per_sample']

    batch_sizes = []
    warnings = []
    for i, vram_bytes in enumerate(gpu_vram_list):
        available = vram_bytes * safety_margin
        if available < fixed:
            warnings.append(
                f"WARNING: GPU {i} has {vram_bytes / (1024**3):.1f} GB VRAM but model "
                f"fixed costs alone require ~{fixed / (1024**3):.1f} GB. "
                f"Training will likely OOM. Consider a smaller model or enabling "
                f"--gradient-checkpointing / --use-8bit-optimizer / --deepspeed."
            )
            bs = 1
        else:
            bs = int((available - fixed) / per_sample) if per_sample > 0 else 1
            bs = max(1, bs)
        if max_batch_size is not None:
            bs = min(bs, max_batch_size)
        batch_sizes.append(bs)

    for w in warnings:
        print(f"\n⚠️  {w}")

    return batch_sizes


def format_bytes(n):
    """Format byte count as human-readable string."""
    if n >= 1024 ** 3:
        return f"{n / (1024**3):.1f} GB"
    elif n >= 1024 ** 2:
        return f"{n / (1024**2):.0f} MB"
    elif n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def format_vram_summary(config, seq_len, gpu_infos, batch_sizes, est, safety_margin=0.85):
    """
    Format a human-readable VRAM estimation summary.

    Args:
        config: BaseMoEConfig
        seq_len: Sequence length
        gpu_infos: List of (name, total_vram_bytes) per GPU
        batch_sizes: List of batch sizes per GPU
        est: Dict from estimate_training_vram (with batch_size=1, for fixed/per_sample)
        safety_margin: Safety margin used

    Returns:
        str: Multi-line summary string
    """
    lines = []
    lines.append(f"VRAM Estimation (seq_len={seq_len}):")
    lines.append(f"  Model params:     {est['n_params']:,}")
    lines.append(f"  Model weights:    {format_bytes(est['model_weights'])}")
    lines.append(f"  Optimizer state:  {format_bytes(est['optimizer_state'])}")
    lines.append(f"  Gradients:        {format_bytes(est['gradients'])}")
    lines.append(f"  CUDA overhead:    {format_bytes(est['cuda_overhead'])}")
    lines.append(f"  Fixed total:      {format_bytes(est['fixed'])}")
    lines.append(f"  Per-sample:       {format_bytes(est['per_sample'])}")
    lines.append("")

    for i, ((name, vram), bs) in enumerate(zip(gpu_infos, batch_sizes)):
        used = est['fixed'] + est['per_sample'] * bs
        pct = used / vram * 100 if vram > 0 else 0
        lines.append(
            f"  GPU {i} ({name}, {format_bytes(vram)}): "
            f"batch_size={bs} (est. {format_bytes(used)} / {format_bytes(vram)}, {pct:.0f}%)"
        )

    return "\n".join(lines)
