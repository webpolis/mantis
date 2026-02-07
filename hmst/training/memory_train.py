"""
Stage 2: Memory System Fine-Tuning

Train episodic and semantic memory systems on long-context retrieval tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from tqdm import tqdm
import os
import random

from hmst.models.base_moe import BaseMoEModel
from hmst.models.ssm import EpisodicMemorySSM
from hmst.memory.episodic import EpisodicMemory
from hmst.memory.semantic import SemanticMemory
from hmst.memory.consolidation import MemoryConsolidator
from hmst.tokenizer import HMSTTokenizer
import numpy as np


def compute_episodic_retrieval_loss(
    ssm_model: EpisodicMemorySSM,
    base_model: BaseMoEModel,
    queries: List[str],
    contexts: List[str],
    tokenizer: HMSTTokenizer,
    device: str
) -> torch.Tensor:
    """
    Train episodic memory to retrieve relevant context given a query.

    Uses contrastive learning: query should match its context, not others.
    """
    batch_size = min(len(queries), 8)  # Small batch for memory efficiency

    # Sample random batch
    indices = random.sample(range(len(queries)), batch_size)
    batch_queries = [queries[i] for i in indices]
    batch_contexts = [contexts[i] for i in indices]

    # Tokenize and embed
    query_tokens = [tokenizer.encode(q)[:128] for q in batch_queries]
    context_tokens = [tokenizer.encode(c)[:512] for c in batch_contexts]

    # Pad sequences
    max_q_len = max((len(q) for q in query_tokens), default=1)
    max_c_len = max((len(c) for c in context_tokens), default=1)

    query_ids = torch.zeros(batch_size, max_q_len, dtype=torch.long, device=device)
    context_ids = torch.zeros(batch_size, max_c_len, dtype=torch.long, device=device)

    for i, (q, c) in enumerate(zip(query_tokens, context_tokens)):
        query_ids[i, :len(q)] = torch.tensor(q, dtype=torch.long)
        context_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)

    # Get embeddings from base model
    with torch.no_grad():
        base_model.eval()
        query_output = base_model(query_ids, return_hidden=True)
        context_output = base_model(context_ids, return_hidden=True)
        query_emb = query_output['last_hidden']  # (batch, seq_len, d_model)
        context_emb = context_output['last_hidden']

    # Compress with SSM
    query_state = ssm_model.encode_sequence(query_emb)  # (batch, d_state)
    context_state = ssm_model.encode_sequence(context_emb)

    # Contrastive loss (InfoNCE)
    # Positive: query matches its own context
    # Negative: query doesn't match other contexts
    similarity = torch.matmul(query_state, context_state.t())  # (batch, batch)
    similarity = similarity / 0.07  # Temperature scaling

    labels = torch.arange(batch_size, device=device)
    loss = F.cross_entropy(similarity, labels)

    return loss


def compute_semantic_embedding_loss(
    base_model: BaseMoEModel,
    facts: List[str],
    tokenizer: HMSTTokenizer,
    device: str
) -> torch.Tensor:
    """
    Train semantic memory embeddings using contrastive learning.

    Similar facts should have similar embeddings.
    """
    if len(facts) < 3:
        return torch.tensor(0.0, device=device)

    # Create triplets: (anchor, positive, negative)
    batch_size = min(len(facts) // 3, 8)

    anchor_texts = []
    positive_texts = []
    negative_texts = []

    for _ in range(batch_size):
        # Ensure we don't go out of bounds
        max_idx = len(facts) - 3
        if max_idx < 0:
            break
        idx = random.randint(0, max_idx)
        anchor_texts.append(facts[idx])
        positive_texts.append(facts[min(idx + 1, len(facts) - 1)])  # Adjacent = related
        negative_texts.append(facts[min(idx + 2, len(facts) - 1)])  # Further = less related

    # Return zero loss if we couldn't create any triplets
    if len(anchor_texts) == 0:
        return torch.tensor(0.0, device=device)

    def embed_texts(texts):
        tokens = [tokenizer.encode(t)[:128] for t in texts]
        max_len = max((len(t) for t in tokens), default=1)
        token_ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        for i, t in enumerate(tokens):
            token_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        with torch.no_grad():
            base_model.eval()
            output = base_model(token_ids, return_hidden=True)
            embeddings = output['last_hidden']  # (batch, seq_len, d_model)

        # Mean pooling and detach to ensure no gradients flow to base model
        return embeddings.mean(dim=1).detach()  # (batch, d_model)

    anchor_emb = embed_texts(anchor_texts)
    positive_emb = embed_texts(positive_texts)
    negative_emb = embed_texts(negative_texts)

    # Triplet margin loss
    loss = F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=0.5)

    return loss


def train_memory_stage(args):
    """
    Entry point for Stage 2: Memory system fine-tuning.

    Called from train.py with --stage 2.

    Args:
        args: Argument namespace from train.py argparse
    """
    print("\n" + "="*80)
    print("Stage 2: Memory Fine-tuning")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load pre-trained base model from checkpoint
    print(f"\nLoading base model from: {args.resume}")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'config' key: {args.resume}")
    config = checkpoint['config']

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer_path}")
    tokenizer = HMSTTokenizer.load(args.tokenizer_path)
    config.base_moe.vocab_size = len(tokenizer)

    # Load base model
    base_model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout,
        load_balance_weight=config.base_moe.load_balance_weight
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(device)
    base_model.eval()  # Freeze base model
    print("✓ Base model loaded successfully (frozen).")

    # 2. Initialize memory systems
    print("Initializing memory systems...")

    # Episodic memory with SSM
    episodic_ssm = EpisodicMemorySSM(
        d_model=config.base_moe.d_model,
        d_state=config.episodic_memory.d_state,
        d_conv=config.episodic_memory.d_conv,
        expand=config.episodic_memory.expand
    ).to(device)

    episodic_memory = EpisodicMemory(
        ssm_model=episodic_ssm,
        max_entries=config.episodic_memory.max_entries,
        device=device
    )

    # Semantic memory (FAISS-based)
    semantic_memory = SemanticMemory(
        dimension=config.semantic_memory.dimension,
        max_entries=config.semantic_memory.max_entries,
        index_type=config.semantic_memory.index_type,
        use_gpu=config.semantic_memory.use_gpu
    )

    print("✓ Memory systems initialized.")

    # 3. Setup optimizer (only for trainable SSM parameters)
    optimizer = torch.optim.AdamW(
        episodic_ssm.parameters(),
        lr=config.training.finetune_lr,
        weight_decay=0.01
    )

    # 4. Prepare training data
    print("\n⚠️  Note: Using demo dataset for memory training.")
    print("   For production, use long-context datasets like QuALITY, NarrativeQA, etc.")
    print("   Dataset should contain query-context pairs and factual knowledge.\n")

    # Demo dataset: query-context pairs for episodic training
    queries = [
        "Who is the main character?",
        "What happens at the end?",
        "Where does the story take place?",
        "What is the central conflict?",
        "How is the problem resolved?"
    ]

    contexts = [
        "The main character is Elizabeth Bennet, a young woman from a middle-class family.",
        "At the end, Elizabeth and Mr. Darcy overcome their pride and prejudice to marry.",
        "The story takes place in rural England during the Regency era.",
        "The central conflict involves class differences and misunderstandings between characters.",
        "The problem is resolved through honest communication and personal growth."
    ]

    # Demo dataset: facts for semantic training
    facts = [
        "The Earth orbits around the Sun.",
        "Planets revolve around stars in elliptical orbits.",
        "Photosynthesis is the process by which plants make food.",
        "Chlorophyll is the green pigment in plants that captures sunlight.",
        "DNA contains genetic information.",
        "Genes are segments of DNA that encode proteins.",
        "Water freezes at 0°C.",
        "Ice is the solid form of water."
    ]

    print(f"✓ Using demo dataset: {len(queries)} query-context pairs, {len(facts)} facts.")

    # 5. Training loop
    print(f"\nStarting memory fine-tuning: {args.epochs} epochs")
    print("="*80 + "\n")

    episodic_ssm.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_steps = args.steps_per_epoch if hasattr(args, 'steps_per_epoch') and args.steps_per_epoch else 100

        pbar = tqdm(range(num_steps), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step in pbar:
            # Episodic memory loss
            episodic_loss = compute_episodic_retrieval_loss(
                episodic_ssm, base_model, queries, contexts, tokenizer, device
            )

            # Semantic memory loss
            semantic_loss = compute_semantic_embedding_loss(
                base_model, facts, tokenizer, device
            )

            # Combined loss
            loss = (
                getattr(config.training, 'episodic_loss_weight', 1.0) * episodic_loss +
                getattr(config.training, 'semantic_loss_weight', 0.5) * semantic_loss
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(episodic_ssm.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'episodic': f'{episodic_loss.item():.4f}',
                'semantic': f'{semantic_loss.item():.4f}'
            })

        # Epoch summary
        avg_loss = epoch_loss / num_steps
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = args.output_dir if hasattr(args, 'output_dir') else os.path.dirname(args.resume)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_path = os.path.join(output_dir, "memory_system_best.pt")
            torch.save({
                'episodic_ssm_state_dict': episodic_ssm.state_dict(),
                'config': config,
                'loss': best_loss
            }, save_path)
            print(f"✓ Best model saved: {save_path}")

    # 6. Final save
    output_dir = args.output_dir if hasattr(args, 'output_dir') else os.path.dirname(args.resume)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_path = os.path.join(output_dir, "memory_system_final.pt")
    torch.save({
        'episodic_ssm_state_dict': episodic_ssm.state_dict(),
        'config': config,
        'loss': best_loss
    }, final_path)

    print(f"\n" + "="*80)
    print(f"✓ Final memory system saved to: {final_path}")
    print("="*80)
    print("\nStage 2: Memory Fine-tuning Complete!")
    print("="*80 + "\n")
