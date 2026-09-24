"""
Evaluation Metrics for MANTIS

Provides metrics for accuracy, hallucination detection, and calibration.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute exact-match accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth answers

    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, target in zip(predictions, targets)
                  if pred.strip().lower() == target.strip().lower())
    return correct / len(predictions)


def token_f1(prediction: str, target: str) -> float:
    """Token-level F1 with multiset overlap (repeated tokens count)."""
    pred_tokens = prediction.lower().split()
    target_tokens = target.lower().split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1_score(predictions: List[str], targets: List[str]) -> float:
    """
    Mean token-level F1 score (useful for generation tasks).

    Args:
        predictions: Model predictions
        targets: Ground truth answers

    Returns:
        F1 score (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")
    if not predictions:
        return 0.0
    return float(np.mean([token_f1(p, t) for p, t in zip(predictions, targets)]))


def compute_hallucination_rate(
    correct: List[bool],
    confidences: List[float],
    threshold: float = 0.8
) -> float:
    """
    Fraction of examples answered incorrectly with confidence >= threshold.

    Args:
        correct: Per-example correctness
        confidences: Model confidence per example (0-1)

    Returns:
        Hallucination rate (0-1)
    """
    if len(correct) != len(confidences):
        raise ValueError("Length mismatch between correctness flags and confidences")
    if not correct:
        return 0.0
    return sum(1 for ok, conf in zip(correct, confidences) if not ok and conf >= threshold) / len(correct)


def compute_calibration_error(
    correct: List[bool],
    confidences: List[float],
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE): how well confidence matches accuracy.

    Args:
        correct: Per-example correctness
        confidences: Confidence scores (0-1)
        n_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error (0-1)
    """
    if len(correct) != len(confidences):
        raise ValueError("Length mismatch")
    if not correct:
        return 0.0

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_indices = np.clip(np.digitize(confidences, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        in_bin = bin_indices == bin_idx
        if in_bin.any():
            ece += in_bin.mean() * abs(confidences[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def compute_perplexity(log_probs: List[float]) -> float:
    """
    Compute perplexity from log probabilities.

    Args:
        log_probs: Log probabilities for each token

    Returns:
        Perplexity score
    """
    if len(log_probs) == 0:
        return float('inf')

    avg_log_prob = np.mean(log_probs)
    return np.exp(-avg_log_prob)


def compute_bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Compute BLEU score for generation quality.

    Simplified BLEU-4 implementation.

    Args:
        predictions: Model predictions
        references: List of reference answers per example

    Returns:
        BLEU score (0-1)
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    total_score = 0.0

    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()

        max_score = 0.0
        for ref in refs:
            ref_tokens = ref.lower().split()

            # Compute precision for n-grams (n=1 to 4)
            precisions = []
            for n in range(1, 5):
                pred_ngrams = get_ngrams(pred_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)

                if len(pred_ngrams) == 0:
                    precisions.append(0.0)
                    continue

                clipped_count = sum(min(pred_ngrams[ng], ref_ngrams[ng])
                                  for ng in pred_ngrams)
                precision = clipped_count / sum(pred_ngrams.values())
                precisions.append(precision)

            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                score = np.exp(np.mean(np.log(precisions)))
            else:
                score = 0.0

            # Brevity penalty (an empty prediction already scored 0)
            if pred_tokens:
                score *= 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))

            max_score = max(max_score, score)

        total_score += max_score

    return total_score / len(predictions) if len(predictions) > 0 else 0.0


def compute_metrics_summary(
    correct: List[bool],
    confidences: List[float] = None
) -> Dict[str, float]:
    """
    Summarize per-example correctness (and confidence, when available).

    Returns:
        Dictionary of metric name -> score
    """
    metrics = {'accuracy': sum(correct) / len(correct) if correct else 0.0}
    if confidences is not None:
        metrics['hallucination_rate'] = compute_hallucination_rate(correct, confidences)
        metrics['calibration_error'] = compute_calibration_error(correct, confidences)
    return metrics
