"""
Stage 2: Memory System Fine-Tuning

Trains the two learnable memory components on (query, context) pairs, with
the Stage 1 base model frozen:

- Episodic SSM: a query's SSM state should match its context's state
  (InfoNCE over in-batch negatives).
- Semantic projection: projected base-model embeddings of a query should
  retrieve its context (InfoNCE over in-batch negatives).

Afterwards every context is written to a semantic memory store with the
trained projection, ready for Stage 3 and inference.
"""

import json
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mantis.memory.semantic import SemanticMemory
from mantis.models.base_moe import BaseMoEModel
from mantis.models.ssm import EpisodicMemorySSM
from mantis.utils.checkpoints import load_base_model

DEMO_PAIRS = [
    ("Who is the main character?",
     "The main character is Elizabeth Bennet, a young woman from a middle-class family."),
    ("What happens at the end?",
     "At the end, Elizabeth and Mr. Darcy overcome their pride and prejudice to marry."),
    ("Where does the story take place?",
     "The story takes place in rural England during the Regency era."),
    ("What is the central conflict?",
     "The central conflict involves class differences and misunderstandings between characters."),
    ("How is the problem resolved?",
     "The problem is resolved through honest communication and personal growth."),
    ("What does the Earth orbit?", "The Earth orbits around the Sun."),
    ("How do plants make food?", "Photosynthesis is the process by which plants make food."),
    ("What carries genetic information?", "DNA contains genetic information."),
]

TEMPERATURE = 0.07


def load_pairs(path: str) -> List[Tuple[str, str]]:
    """JSONL with {"query": ..., "context": ...} per line."""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                pairs.append((record['query'], record['context']))
    return pairs


@torch.no_grad()
def embed_batch(base_model: BaseMoEModel, tokenizer, texts: List[str], max_len: int, device: str):
    """
    Right-padded frozen base-model hidden states.

    Returns:
        hidden: (batch, seq_len, d_model)
        mask: (batch, seq_len), 1 = real token
    """
    token_lists = [tokenizer.encode(t)[:max_len] or [tokenizer.eos_token_id] for t in texts]
    length = max(len(t) for t in token_lists)
    ids = torch.full((len(texts), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros(len(texts), length, device=device)
    for i, tokens in enumerate(token_lists):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = 1
    return base_model(ids, return_hidden=True)['last_hidden'], mask


def pooled(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the sequence (matches BaseMoEModel.encode)."""
    m = mask.unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def info_nce(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Query i should match key i among all keys in the batch."""
    logits = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).t() / TEMPERATURE
    return F.cross_entropy(logits, torch.arange(len(queries), device=queries.device))


def train_memory_stage(args):
    """
    Entry point for Stage 2: Memory system fine-tuning.

    Called from train.py with --stage 2. The optional positional data file is
    JSONL with {"query": ..., "context": ...} records.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nLoading base model from: {args.resume}")
    base_model, tokenizer, checkpoint = load_base_model(args.resume, device, args.tokenizer_path)
    base_model.requires_grad_(False)
    config = checkpoint['config']
    d_model = config.base_moe.d_model
    print("✓ Base model loaded (frozen).")

    em = config.episodic_memory
    episodic_ssm = EpisodicMemorySSM(
        d_model=d_model, d_state=em.d_state, n_blocks=em.n_blocks, d_conv=em.d_conv,
        expand=em.expand, max_seq_len=em.max_seq_len, dropout=em.dropout,
    ).to(device)
    projection = nn.Linear(d_model, d_model).to(device)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(d_model))
        projection.bias.zero_()

    lr = args.learning_rate or config.training.finetune_lr
    optimizer = torch.optim.AdamW(
        list(episodic_ssm.parameters()) + list(projection.parameters()), lr=lr, weight_decay=0.01
    )

    if args.train_file:
        pairs = load_pairs(args.train_file)
        print(f"✓ Loaded {len(pairs)} query-context pairs from {args.train_file}")
    else:
        pairs = list(DEMO_PAIRS)
        print("⚠️  No data file given: using the 8-pair demo dataset.")
        print("   Pass a JSONL file of {\"query\": ..., \"context\": ...} records (e.g. from QuALITY, NarrativeQA).")
    if len(pairs) < 2:
        raise ValueError("Contrastive training needs at least 2 query-context pairs")

    max_len = base_model.max_seq_len
    # In-batch negatives need at least two pairs per batch
    batch_size = max(2, min(args.batch_size, len(pairs)))
    num_steps = args.steps_per_epoch or 100
    episodic_weight = config.training.episodic_loss_weight
    semantic_weight = config.training.semantic_loss_weight

    def save(path, loss):
        torch.save({
            'episodic_ssm_state_dict': episodic_ssm.state_dict(),
            'semantic_projection_state_dict': projection.state_dict(),
            'config': config,
            'tokenizer_fingerprint': tokenizer.fingerprint(),
            'base_checkpoint': os.path.abspath(args.resume),
            'loss': loss,
        }, path)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nStarting memory fine-tuning: {args.epochs} epochs × {num_steps} steps, batch {batch_size}, lr {lr}")
    print("=" * 80 + "\n")

    episodic_ssm.train()
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(num_steps), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in pbar:
            batch = random.sample(pairs, batch_size)
            q_hidden, q_mask = embed_batch(base_model, tokenizer, [q for q, _ in batch], max_len, device)
            c_hidden, c_mask = embed_batch(base_model, tokenizer, [c for _, c in batch], max_len, device)

            episodic_loss = info_nce(
                episodic_ssm.encode_sequence(q_hidden, q_mask),
                episodic_ssm.encode_sequence(c_hidden, c_mask),
            )
            semantic_loss = info_nce(projection(pooled(q_hidden, q_mask)), projection(pooled(c_hidden, c_mask)))
            loss = episodic_weight * episodic_loss + semantic_weight * semantic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(episodic_ssm.parameters()) + list(projection.parameters()), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'episodic': f'{episodic_loss.item():.4f}',
                              'semantic': f'{semantic_loss.item():.4f}'})

        avg_loss = epoch_loss / num_steps
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save(os.path.join(args.output_dir, "memory_system_best.pt"), best_loss)
            print("✓ Best memory system saved")

    final_path = os.path.join(args.output_dir, "memory_system_final.pt")
    save(final_path, best_loss)

    # Build the semantic store from all contexts with the trained projection
    episodic_ssm.eval()
    projection.eval()
    semantic_memory = SemanticMemory(
        dimension=config.semantic_memory.dimension,
        max_entries=config.semantic_memory.max_entries,
        index_type=config.semantic_memory.index_type,
        use_gpu=config.semantic_memory.use_gpu,
        projection=projection,
    )
    contexts = list(dict.fromkeys(c for _, c in pairs))
    for i in range(0, len(contexts), 32):
        chunk = contexts[i:i + 32]
        hidden, mask = embed_batch(base_model, tokenizer, chunk, max_len, device)
        semantic_memory.add_batch(pooled(hidden, mask), chunk)
    store_path = os.path.join(args.output_dir, "semantic_memory")
    semantic_memory.save(store_path)

    print(f"\n{'='*80}")
    print(f"✓ Memory system saved to: {final_path}")
    print(f"✓ Semantic store ({semantic_memory.size()} facts) saved to: {store_path}")
    print(f"  Stage 3: --memory-checkpoint {final_path} --semantic-store {store_path}")
    print("=" * 80 + "\n")
