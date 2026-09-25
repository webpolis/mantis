"""
Stage 4: Critic Training

Supervised training of the verification critic on labeled
(query, response, evidence) triples. The critic reads the frozen Stage 1
model's final hidden states, so it starts from the backbone's language
knowledge rather than from scratch. The trained critic enables the
verification gate in Stage 3 and in the full inference engine.

Data: JSONL with {"query": str, "response": str, "evidence": str or [str]
(optional), "label": 0 or 1} per line, where 1 means the response is correct
given the evidence.

The data is split train / calibration / validation. Temperature scaling is
fitted on the calibration split so the deployed score is a probability;
validation reports loss, accuracy, Brier score and ECE.
"""

import json
import os
import random
import re
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from mantis.models.critic import CriticModel
from mantis.training.rl_train import DEMO_QA
from mantis.utils.checkpoints import load_base_model, model_fingerprint


def load_examples(path: str) -> List[Dict]:
    examples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                evidence = record.get('evidence')
                if isinstance(evidence, list):
                    evidence = "\n".join(evidence)
                examples.append({
                    'query': record['query'],
                    'response': record['response'],
                    'evidence': evidence,
                    'label': float(record['label']),
                })
    return examples


def negate(answer: str) -> str:
    """A near-miss negative: the answer with a negation inserted."""
    match = re.search(r"\b(is|are|was|were|has|have|does|do|can|will)\b", answer)
    if match:
        return answer[:match.end()] + " not" + answer[match.end():]
    return "It is not true that " + answer[0].lower() + answer[1:]


def perturb(answer: str) -> str:
    """A near-miss negative: a number changed, else the first capitalized word altered."""
    if re.search(r"\d", answer):
        return re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), answer, count=1)
    words = answer.split()
    for i, word in enumerate(words[1:], 1):
        if word[:1].isupper():
            words[i] = word[:-1] + ("a" if word[-1] != "a" else "o")
            return " ".join(words)
    return "Un" + answer[0].lower() + answer[1:]


def demo_examples() -> List[Dict]:
    """
    Correct answers labeled 1; answers to other questions, negated answers
    and perturbed answers labeled 0.
    """
    examples = []
    for i, (query, answer) in enumerate(DEMO_QA):
        wrong = DEMO_QA[(i + 1) % len(DEMO_QA)][1]
        examples.append({'query': query, 'response': answer, 'evidence': answer, 'label': 1.0})
        examples.append({'query': query, 'response': wrong, 'evidence': answer, 'label': 0.0})
        examples.append({'query': query, 'response': negate(answer), 'evidence': answer, 'label': 0.0})
        examples.append({'query': query, 'response': perturb(answer), 'evidence': answer, 'label': 0.0})
    return examples


def features(critic: CriticModel, base_model, tokenizer, batch: List[Dict], device: str):
    """Tokenize, budget-truncate, right-pad and encode a batch with the frozen base model."""
    rows = [
        critic.build_input(
            tokenizer.encode(ex['query']),
            tokenizer.encode(ex['response']),
            tokenizer.encode(ex['evidence']) if ex['evidence'] else None,
        )
        for ex in batch
    ]
    input_ids, segment_ids, attention_mask = CriticModel.collate(rows, tokenizer.pad_token_id, device)
    hidden = CriticModel.backbone_features(base_model, input_ids)
    labels = torch.tensor([[ex['label']] for ex in batch], device=device)
    return hidden, segment_ids, attention_mask, labels


def calibration_metrics(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> Tuple[float, float]:
    """(Brier score, expected calibration error) of probabilities against {0, 1} labels."""
    probs, labels = probs.flatten().float(), labels.flatten().float()
    brier = ((probs - labels) ** 2).mean().item()
    bins = torch.clamp((probs * n_bins).long(), max=n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(probs[in_bin].mean().item() - labels[in_bin].mean().item())
    return brier, ece


def train_critic_stage(args):
    """Entry point for Stage 4, called from train.py with --stage 4."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model, tokenizer, checkpoint = load_base_model(args.resume, device, args.tokenizer_path)
    base_model.requires_grad_(False)
    fingerprint = model_fingerprint(base_model, tokenizer)
    config = checkpoint['config']
    critic = CriticModel.from_config(config.critic, config.base_moe).to(device)

    if args.train_file:
        examples = load_examples(args.train_file)
        print(f"✓ Loaded {len(examples)} labeled examples from {args.train_file}")
    else:
        examples = demo_examples()
        print(f"⚠️  No data file given: using a {len(examples)}-example demo dataset.")
        print('   Pass a JSONL file of {"query", "response", "evidence", "label"} records for real training.')

    random.Random(42).shuffle(examples)
    n_held = max(1, len(examples) // 10)
    val_set, calib_set, train_set = examples[:n_held], examples[n_held:2 * n_held], examples[2 * n_held:]
    if not train_set:
        raise ValueError("Critic training needs at least 3 examples (train, calibration and validation)")

    lr = args.learning_rate or config.training.critic_lr
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)

    def save(path, val_metrics):
        torch.save({
            'critic_state_dict': critic.state_dict(),
            'config': config,
            'tokenizer_fingerprint': tokenizer.fingerprint(),
            'embedding_fingerprint': fingerprint,
            'val_metrics': val_metrics,
        }, path)

    @torch.no_grad()
    def predict(dataset):
        """(logits, labels) over a split, in eval mode."""
        critic.eval()
        logits, labels = [], []
        for i in range(0, len(dataset), args.batch_size):
            hidden, segment_ids, mask, batch_labels = features(critic, base_model, tokenizer,
                                                               dataset[i:i + args.batch_size], device)
            logits.append(critic(hidden, segment_ids, mask))
            labels.append(batch_labels)
        return torch.cat(logits), torch.cat(labels)

    def evaluate():
        logits, labels = predict(val_set)
        probs = critic.probability(logits)
        brier, ece = calibration_metrics(probs, labels)
        return {
            'loss': critic.compute_loss(logits, labels).item(),
            'accuracy': ((probs > 0.5).float() == labels).float().mean().item(),
            'brier': brier,
            'ece': ece,
        }

    print(f"\nTraining critic: {len(train_set)} train / {len(calib_set)} calibration / {len(val_set)} val, "
          f"{args.epochs} epochs, lr {lr}")
    best_val, best_state = float('inf'), None
    for epoch in range(args.epochs):
        critic.train()
        random.shuffle(train_set)
        epoch_loss = 0.0
        batches = range(0, len(train_set), args.batch_size)
        for i in tqdm(batches, desc=f"Epoch {epoch+1}/{args.epochs}"):
            hidden, segment_ids, mask, labels = features(critic, base_model, tokenizer,
                                                         train_set[i:i + args.batch_size], device)
            loss = critic.compute_loss(critic(hidden, segment_ids, mask), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        metrics = evaluate()
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss / len(batches):.4f}, "
              f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.2%}")
        if metrics['loss'] < best_val:
            best_val = metrics['loss']
            best_state = {k: v.detach().clone() for k, v in critic.state_dict().items()}
            print("✓ Best epoch so far")

    critic.load_state_dict(best_state)
    logits, labels = predict(calib_set)
    temperature = critic.fit_temperature(logits, labels)
    metrics = evaluate()
    print(f"\nCalibration: temperature {temperature:.3f} fitted on {len(calib_set)} examples")
    print(f"Validation: loss {metrics['loss']:.4f}, accuracy {metrics['accuracy']:.2%}, "
          f"Brier {metrics['brier']:.4f}, ECE {metrics['ece']:.4f}")

    best_path = os.path.join(args.output_dir, "critic_best.pt")
    save(best_path, metrics)
    save(os.path.join(args.output_dir, "critic_final.pt"), metrics)
    print(f"\n✓ Critic saved to: {best_path}")
    print(f"  Stage 3: --critic-checkpoint {best_path}")
