"""
Stage 4: Critic Training

Supervised training of the hallucination critic on labeled
(query, response, facts) triples. The trained critic enables the
verification gate in Stage 3 and in the full inference engine.

Data: JSONL with {"query": str, "response": str, "facts": str (optional),
"label": 0 or 1} per line, where 1 means the response is correct.
"""

import json
import os
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from mantis.models.critic import CriticModel
from mantis.training.rl_train import DEMO_QA
from mantis.utils.checkpoints import compat_load, load_tokenizer


def load_examples(path: str) -> List[Dict]:
    examples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                examples.append({
                    'query': record['query'],
                    'response': record['response'],
                    'facts': record.get('facts'),
                    'label': float(record['label']),
                })
    return examples


def demo_examples() -> List[Dict]:
    """Correct answers labeled 1, answers to other questions labeled 0."""
    examples = []
    for i, (query, answer) in enumerate(DEMO_QA):
        wrong = DEMO_QA[(i + 1) % len(DEMO_QA)][1]
        examples.append({'query': query, 'response': answer, 'facts': answer, 'label': 1.0})
        examples.append({'query': query, 'response': wrong, 'facts': answer, 'label': 0.0})
    return examples


def collate(critic: CriticModel, tokenizer, batch: List[Dict], device: str):
    """Tokenize, budget-truncate and right-pad a batch."""
    rows = [
        critic.build_input(
            tokenizer.encode(ex['query']),
            tokenizer.encode(ex['response']),
            tokenizer.encode(ex['facts']) if ex['facts'] else None,
        )
        for ex in batch
    ]
    length = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    segment_ids = torch.zeros(len(rows), length, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), length, dtype=torch.long)
    for i, (ids, segs) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        segment_ids[i, :len(segs)] = torch.tensor(segs)
        attention_mask[i, :len(ids)] = 1
    labels = torch.tensor([[ex['label']] for ex in batch])
    return input_ids.to(device), segment_ids.to(device), attention_mask.to(device), labels.to(device)


def train_critic_stage(args):
    """Entry point for Stage 4, called from train.py with --stage 4."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = compat_load(args.resume)
    tokenizer = load_tokenizer(args.resume, checkpoint, args.tokenizer_path)
    config = checkpoint['config']
    config.critic.vocab_size = len(tokenizer)

    critic = CriticModel(**vars(config.critic)).to(device)

    if args.train_file:
        examples = load_examples(args.train_file)
        print(f"✓ Loaded {len(examples)} labeled examples from {args.train_file}")
    else:
        examples = demo_examples()
        print("⚠️  No data file given: using a 20-example demo dataset.")
        print('   Pass a JSONL file of {"query", "response", "facts", "label"} records for real training.')

    random.Random(42).shuffle(examples)
    n_val = max(1, len(examples) // 10)
    train_set, val_set = examples[n_val:], examples[:n_val]
    if not train_set:
        raise ValueError("Critic training needs at least 2 examples")

    lr = args.learning_rate or config.training.critic_lr
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)

    def save(path, val_loss):
        torch.save({
            'critic_state_dict': critic.state_dict(),
            'config': config,
            'tokenizer_fingerprint': tokenizer.fingerprint(),
            'val_loss': val_loss,
        }, path)

    @torch.no_grad()
    def evaluate():
        critic.eval()
        total, correct = 0.0, 0
        for i in range(0, len(val_set), args.batch_size):
            batch = val_set[i:i + args.batch_size]
            input_ids, segment_ids, mask, labels = collate(critic, tokenizer, batch, device)
            correctness, confidence = critic(input_ids, segment_ids, mask)
            total += critic.compute_loss(correctness, confidence, labels).item() * len(batch)
            correct += ((correctness > 0.5).float() == labels).sum().item()
        critic.train()
        return total / len(val_set), correct / len(val_set)

    print(f"\nTraining critic: {len(train_set)} train / {len(val_set)} val, {args.epochs} epochs, lr {lr}")
    best_val = float('inf')
    for epoch in range(args.epochs):
        critic.train()
        random.shuffle(train_set)
        epoch_loss = 0.0
        batches = range(0, len(train_set), args.batch_size)
        for i in tqdm(batches, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = train_set[i:i + args.batch_size]
            input_ids, segment_ids, mask, labels = collate(critic, tokenizer, batch, device)
            correctness, confidence = critic(input_ids, segment_ids, mask)
            loss = critic.compute_loss(correctness, confidence, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        val_loss, val_acc = evaluate()
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss / len(batches):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        if val_loss < best_val:
            best_val = val_loss
            save(os.path.join(args.output_dir, "critic_best.pt"), val_loss)
            print("✓ Best critic saved")

    final_path = os.path.join(args.output_dir, "critic_final.pt")
    save(final_path, best_val)
    print(f"\n✓ Critic saved to: {final_path}")
    print(f"  Stage 3: --critic-checkpoint {os.path.join(args.output_dir, 'critic_best.pt')}")
