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


def compute_f1_score(predictions: List[str], targets: List[str]) -> float:
    """
    Compute token-level F1 score (useful for generation tasks).

    Args:
        predictions: Model predictions
        targets: Ground truth answers

    Returns:
        F1 score (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")

    f1_scores = []

    for pred, target in zip(predictions, targets):
        pred_tokens = set(pred.lower().split())
        target_tokens = set(target.lower().split())

        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            f1_scores.append(1.0)
            continue
        elif len(pred_tokens) == 0 or len(target_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & target_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(target_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)

    return np.mean(f1_scores) if f1_scores else 0.0


def compute_hallucination_rate(
    predictions: List[str],
    targets: List[str],
    confidences: List[float]
) -> float:
    """
    Compute hallucination rate (incorrect predictions with high confidence).

    Args:
        predictions: Model predictions
        targets: Ground truth answers
        confidences: Confidence scores (0-1)

    Returns:
        Hallucination rate (0-1)
    """
    if len(predictions) != len(targets) or len(predictions) != len(confidences):
        raise ValueError("Length mismatch between predictions, targets, and confidences")

    high_confidence_threshold = 0.8
    hallucinations = 0

    for pred, target, conf in zip(predictions, targets, confidences):
        is_correct = pred.strip().lower() == target.strip().lower()
        is_high_confidence = conf >= high_confidence_threshold

        if not is_correct and is_high_confidence:
            hallucinations += 1

    return hallucinations / len(predictions) if len(predictions) > 0 else 0.0


def compute_calibration_error(
    predictions: List[str],
    targets: List[str],
    confidences: List[float],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well confidence scores match actual accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth answers
        confidences: Confidence scores (0-1)
        n_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error (0-1)
    """
    if len(predictions) != len(targets) or len(predictions) != len(confidences):
        raise ValueError("Length mismatch")

    if len(predictions) == 0:
        return 0.0

    # Bin samples by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0

    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        bin_size = np.sum(bin_mask)

        if bin_size == 0:
            continue

        # Average confidence in bin
        bin_confidence = np.mean([conf for conf, in_bin in zip(confidences, bin_mask) if in_bin])

        # Accuracy in bin
        bin_predictions = [pred for pred, in_bin in zip(predictions, bin_mask) if in_bin]
        bin_targets = [target for target, in_bin in zip(targets, bin_mask) if in_bin]
        bin_accuracy = compute_accuracy(bin_predictions, bin_targets)

        # Weighted calibration error
        ece += (bin_size / len(predictions)) * abs(bin_confidence - bin_accuracy)

    return ece


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
    from collections import Counter

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

            # Brevity penalty
            bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
            score *= bp

            max_score = max(max_score, score)

        total_score += max_score

    return total_score / len(predictions) if len(predictions) > 0 else 0.0


def compute_metrics_summary(
    predictions: List[str],
    targets: List[str],
    confidences: List[float] = None
) -> Dict[str, float]:
    """
    Compute all metrics and return a summary.

    Args:
        predictions: Model predictions
        targets: Ground truth answers
        confidences: Optional confidence scores

    Returns:
        Dictionary of metric name -> score
    """
    metrics = {
        'accuracy': compute_accuracy(predictions, targets),
        'f1_score': compute_f1_score(predictions, targets)
    }

    if confidences is not None:
        metrics['hallucination_rate'] = compute_hallucination_rate(
            predictions, targets, confidences
        )
        metrics['calibration_error'] = compute_calibration_error(
            predictions, targets, confidences
        )

    return metrics
