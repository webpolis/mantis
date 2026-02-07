# HMST Architecture & Implementation Review

Systematic review of the model architecture, training pipeline, inference engine, and
memory systems. Issues are categorized by severity.

---

## Critical Issues (Will Crash at Runtime)

### 1. Semantic Memory Dimension Mismatch with Base Model

**Files**: `hmst/configs/model_config.py`, `hmst/inference/engine.py`

The default config sets `BaseMoEConfig.d_model = 2048` but
`SemanticMemoryConfig.dimension = 1536`. When the inference engine passes a query
embedding from the base model to semantic memory retrieval (`engine.py:129`):

```python
facts = self.semantic.retrieve(query_emb.squeeze(0), top_k=5)
```

`query_emb` is shape `(d_model,)` = `(2048,)`, but FAISS expects a vector of
`dimension=1536`. This will cause a FAISS dimension mismatch error. There is no
projection layer between the base model output and the semantic memory index.

**Fix**: Either make `SemanticMemoryConfig.dimension` match `d_model`, or add a
learned projection layer in the inference engine to map `d_model -> dimension`.

---

### 2. Memory Training Accesses Non-Existent Dict Key

**File**: `hmst/training/memory_train.py:64-65, 128`

```python
query_output = base_model(query_ids)
query_emb = query_output['embeddings']  # KeyError: 'embeddings'
```

`BaseMoEModel.forward()` returns a dict with keys `'logits'` and
`'load_balance_loss'` (and optionally `'hidden_states'`/`'last_hidden'` when
`return_hidden=True`). There is no `'embeddings'` key. Stage 2 memory training will
crash immediately.

**Fix**: Use `base_model(query_ids, return_hidden=True)['last_hidden']` or call
`base_model.encode(query_ids)`.

---

### 3. Meta-Controller Input Dimension Mismatch (Default Config)

**Files**: `hmst/configs/model_config.py`, `hmst/inference/engine.py:98`,
`hmst/models/meta_controller.py:41`

The MetaController's input projection is:

```python
self.embedding = nn.Linear(d_model + state_dim, d_model)
```

where `d_model` is `MetaControllerConfig.d_model = 1024` (default). But the inference
engine feeds it `query_emb` from `BaseMoEModel.encode()`, which has dimension
`BaseMoEConfig.d_model = 2048`. So the actual input dimension is `2048 + 128 = 2176`,
but the linear layer expects `1024 + 128 = 1152`. This causes a shape mismatch.

The RL training script (`rl_train.py:482`) works around this by constructing the
MetaController with `d_model=config.base_moe.d_model`, but the default
`MetaControllerConfig` and the `HMSTInferenceEngine` don't enforce this alignment.

**Fix**: The MetaController's `d_model` should be set to the base model's `d_model` by
default, or a projection layer should be added.

---

### 4. No Sequence Length Bounds Check in Autoregressive Decoding

**File**: `hmst/inference/engine.py:322-337`

```python
for _ in range(max_length):
    output = self.base(generated, expert_weights=expert_weights)
```

The `generated` tensor grows by 1 token per iteration. When it exceeds
`max_seq_len=8192`, the position embedding lookup in `BaseMoEModel.forward()` will
raise an `IndexError`:

```python
positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
x = self.token_embedding(input_ids) + self.position_embedding(positions)
# position_embedding is nn.Embedding(max_seq_len, d_model)
```

The same issue exists in `inference.py:266` (`InferenceEngine._generate()`).

**Fix**: Clamp or truncate `generated` to the last `max_seq_len` tokens, or implement
a KV-cache to avoid reprocessing the full sequence.

---

### 5. Memory Consolidation: Dict-with-Tensor Comparison Fails

**File**: `hmst/memory/consolidation.py:129-132`

```python
importance = sum(
    self.episodic.importance_scores[i]
    for i, e in enumerate(self.episodic.entries)
    if e in group
) / len(group)
```

The `e in group` check compares dicts containing PyTorch tensors. Python's `in`
operator calls `__eq__` on each element, and for dicts containing tensors,
`dict.__eq__` compares tensor values using `tensor.__eq__`, which returns a tensor
rather than a boolean. This will raise:

```
RuntimeError: Boolean value of Tensor with more than one element is ambiguous
```

**Fix**: Compare by a unique identifier (e.g., `timestamp`) instead of dict equality.

---

### 6. Inference Engine: Division by Zero at Temperature=0

**File**: `hmst/inference/engine.py:361`

```python
logits = logits / temperature
```

The `HMSTInferenceEngine._sample()` method has no guard for `temperature=0`. The CLI
wrapper (`inference.py:232-233`) catches this, but direct use of the engine class will
crash with a division-by-zero or produce `inf` values.

**Fix**: Add a guard in `_sample()` to use argmax (greedy) when `temperature <= 0`.

---

## Serious Logic Issues (Incorrect Behavior)

### 7. PAD Token Aliased to EOS Token Breaks Training Signal

**File**: `hmst/tokenizer.py:32-33`, `train.py:788`

```python
# tokenizer.py
self.tokenizer.pad_token = self.tokenizer.eos_token  # pad_token_id == eos_token_id

# train.py
lm_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=tokenizer.pad_token_id  # This also ignores EOS tokens!
)
```

Since `pad_token_id == eos_token_id`, every EOS token in the labels is treated as
padding and excluded from the loss. The model never receives gradient signal to predict
EOS, so it cannot learn when to stop generating. This will manifest as the model
generating until `max_length` is hit every time.

**Fix**: Assign a distinct pad token ID (e.g., add a `<pad>` token to the vocabulary)
or use a different `ignore_index` strategy that doesn't conflict with EOS.

---

### 8. PPO Uses Simple Advantage, Not GAE as Specified

**File**: `hmst/training/rl_train.py:244-246`

```python
values = self.value_net(states).squeeze(-1)
advantages = rewards - values.detach()
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

The `__init__` stores `self.discount` (gamma=0.99) and `self.gae_lambda` (0.95), but
they are never used. The actual advantage computation is a simple reward-minus-value
baseline, which has much higher variance than GAE. The proper GAE formula is:

```
A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

Additionally, the PPO training treats each query as a single-step episode
(`collect_episode` stores one transition per query), making the discount and GAE
parameters conceptually inapplicable. If single-step episodes are intended, the
advantage simplifies to `r - V(s)`, which is what the code does -- but then
`discount` and `gae_lambda` are misleading dead parameters.

**Fix**: Either implement proper GAE for multi-step episodes, or remove the unused
`discount`/`gae_lambda` parameters to avoid confusion.

---

### 9. Redundant Double Forward Pass in Inference Engine

**File**: `hmst/inference/engine.py:80-86`

```python
# First forward pass (full model + mean pooling)
query_emb = self.base.encode(query_tokens)  # calls self.forward(return_hidden=True)

# Second forward pass (identical)
with torch.no_grad():
    output = self.base(query_tokens, return_hidden=True)
    logits = output['logits']
    hidden = output['last_hidden']
```

The `encode()` method internally calls `self.forward(input_ids, return_hidden=True)`
and then mean-pools the last hidden state. The very next block calls `forward()` again
with the same input. This doubles the compute cost of every query.

**Fix**: Call `forward(return_hidden=True)` once and derive both `query_emb` (via mean
pooling `last_hidden`) and `logits` from the same output.

---

### 10. `get_large_config()` Doesn't Sync n_experts Across Components

**File**: `hmst/configs/model_config.py:213-221`

```python
def get_large_config() -> HMSTConfig:
    config = HMSTConfig()
    config.base_moe.n_experts = 16
    # Missing: config.meta_controller.n_experts = 16
    return config
```

The base model is configured with 16 experts, but the meta-controller retains the
default `n_experts=8`. The meta-controller's expert selector output has 8 dimensions
while the MoE layers expect 16-dimensional expert weight vectors. This would cause a
dimension mismatch when using the `get_large_config()` preset.

The same issue exists in `get_small_config()` for `n_heads` -- the base model changes
`n_heads` but `MetaControllerConfig.n_heads` stays at 16 by default (which is fine
since they're independent), but `n_experts` must match.

**Fix**: Sync `meta_controller.n_experts` with `base_moe.n_experts` in all config
presets (already done in `get_micro_config`, `get_tiny_config`, `get_small_config`,
but missing from `get_large_config` and `get_extmem_config`).

---

### 11. Load Balancing Loss Uses Suboptimal Formulation

**File**: `hmst/models/base_moe.py:124-135`

```python
def _compute_load_balance_loss(self, gate_probs):
    expert_freq = gate_probs.mean(dim=0)  # Mean routing probability per expert
    target = 1.0 / self.n_experts
    balance_loss = ((expert_freq - target) ** 2).sum()
    return self.load_balance_weight * balance_loss
```

This computes the squared deviation of mean routing probabilities from uniform. The
standard Switch Transformer formulation is:

```
L_balance = N * sum_i(f_i * P_i)
```

where `f_i` is the fraction of tokens *dispatched* to expert i (hard routing), and
`P_i` is the mean routing *probability* for expert i. The current formulation only
penalizes the soft routing probabilities, not the hard dispatch counts. When top-k
routing creates skewed dispatch despite balanced probabilities (which happens in
practice), this loss won't correct it.

---

### 12. Validation Loss Aggregation Is Mathematically Incorrect for Multi-GPU

**File**: `train.py:822-825`

```python
val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
val_loss_tensor = accelerator.gather(val_loss_tensor).mean()
```

Each GPU computes its own average loss over its local validation batches via
`total_loss / total_tokens`. If GPUs process different numbers of tokens (e.g., due to
uneven data distribution), taking the mean of per-GPU averages is not equal to the
global average. The correct approach is to gather both `total_loss` and `total_tokens`
separately and compute `sum(losses) / sum(tokens)`.

---

## Minor Issues & Design Concerns

### 13. Critic Model Pools from First Token Without CLS Token

**File**: `hmst/models/critic.py:119`

```python
pooled = encoded[:, 0, :]  # (batch, d_model)
```

The comment mentions "[CLS] token" but the model doesn't prepend a special CLS token.
The first position is just the first query token. Without a dedicated CLS embedding
trained for classification, this is equivalent to using an arbitrary token's
representation for the classification decision. Mean pooling (which is used as the
alternative strategy in the comment) would be more robust for an untrained model.

---

### 14. Episodic Memory Retrieval Uses Two Different Projection Paths

**File**: `hmst/memory/episodic.py:99-108`

```python
if query_embedding.dim() == 1:
    query_embedding = query_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    _, query_state = self.ssm(query_embedding, return_state=True)
    query_embedding = query_state.squeeze(0)  # (d_state,)
else:
    query_embedding = query_embedding.unsqueeze(0)  # (1, d_model)
    query_embedding = self.ssm.state_proj(query_embedding).squeeze(0)  # (d_state,)
```

The 1D case runs the full SSM stack (8 Mamba blocks + mean pool + projection), while
the 2D case uses only the final linear projection (`state_proj`). These produce
fundamentally different embeddings from the same input, making retrieval quality
depend on the shape of the input tensor rather than its content.

---

### 15. PPO Value Loss Is Not Clipped

**File**: `hmst/training/rl_train.py:278-279`

```python
values = self.value_net(states).squeeze(-1)
value_loss = ((values - rewards) ** 2).mean()
```

Standard PPO clips the value function loss to prevent large updates:

```python
v_clipped = old_values + torch.clamp(values - old_values, -epsilon, epsilon)
value_loss = max((values - returns)^2, (v_clipped - returns)^2).mean()
```

The current implementation allows unbounded value function updates, which can
destabilize training when rewards have high variance.

---

### 16. Semantic Memory `retrieve_with_metadata` Uses O(n) Linear Search

**File**: `hmst/memory/semantic.py:248-259`

```python
for text, dist in zip(texts, distances):
    for entry in self.metadata:  # O(n) search
        if entry['text'] == text:
            results.append({...})
            break
```

This is O(k*n) where n is the total number of entries. For the intended scale of 1M+
entries, this is a performance bottleneck. FAISS returns indices that could be used
directly to index into `self.metadata`.

---

### 17. No KV-Cache in Autoregressive Decoding

**Files**: `hmst/inference/engine.py:320-337`, `inference.py:240-266`

Both inference paths recompute attention for the entire sequence at every step:

```python
for step in range(max_length):
    output = self.model(input_tensor)  # Full seq_len attention each step
```

This is O(n^2 * max_length) total compute instead of O(n * max_length) with a
KV-cache. For a model designed for 8K context, generation will be extremely slow.

---

### 18. `HuggingFaceDataset` Streaming Mode Ignores `idx` Parameter

**File**: `train.py:313-330`

The `__getitem__` method for streaming mode ignores the `idx` argument entirely and
returns the next item from a sequential iterator. This only works because the code
sets `shuffle=False` and `num_workers=0` for streaming mode. Any future change to
those settings would silently produce incorrect data ordering.

---

### 19. Deprecated `torch.cuda.amp.GradScaler` Usage

**File**: `hmst/training/pretrain.py:100`

```python
self.scaler = torch.cuda.amp.GradScaler()
```

and:

```python
with torch.cuda.amp.autocast(enabled=self.scaler is not None):
```

These are deprecated in recent PyTorch versions in favor of
`torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`. This won't break
currently but will emit deprecation warnings and may break in future PyTorch versions.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| **Critical (crash)** | 6 | Dimension mismatches, missing dict keys, unbounded sequences, tensor comparison, div-by-zero |
| **Serious (wrong behavior)** | 6 | PAD/EOS conflation, fake GAE, redundant forward pass, config desync, bad loss formulation, multi-GPU aggregation |
| **Minor** | 7 | No KV-cache, linear search, deprecated APIs, inconsistent projection paths |

The most impactful issues to fix first are #1 (semantic dimension mismatch), #2
(memory training KeyError), #7 (PAD=EOS), and #4 (sequence length bounds), as these
affect the core training and inference loops.
