"""
Evaluation Metrics for MANTIS

Correctness, abstention bookkeeping, confident-error rate and calibration.
Metric names say what they measure: `confident_error_rate` is the fraction
of all examples answered wrongly with high confidence, not a "hallucination
rate"; `error_rate` is the plain fraction of wrong answers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

from mantis.inference.prompting import word_f1

token_f1 = word_f1


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


def _check_lengths(correct: List[bool], other: Optional[List], name: str) -> None:
    if other is not None and len(correct) != len(other):
        raise ValueError(f"Length mismatch between correctness flags and {name}")


def error_rate(correct: List[bool]) -> float:
    """Fraction of all examples answered incorrectly (abstentions count as incorrect)."""
    if not correct:
        return 0.0
    return sum(1 for ok in correct if not ok) / len(correct)


def coverage(abstained: List[bool]) -> float:
    """Fraction of examples the system answered (did not abstain on)."""
    if not abstained:
        return 0.0
    return sum(1 for a in abstained if not a) / len(abstained)


def answered_error_rate(correct: List[bool], abstained: List[bool]) -> float:
    """Fraction of answered examples that are wrong (the risk at the achieved coverage)."""
    _check_lengths(correct, abstained, "abstention flags")
    answered = [ok for ok, a in zip(correct, abstained) if not a]
    if not answered:
        return 0.0
    return sum(1 for ok in answered if not ok) / len(answered)


def confident_error_rate(
    correct: List[bool],
    confidences: List[float],
    threshold: float = 0.8
) -> float:
    """
    Fraction of all examples answered incorrectly with confidence >= threshold.

    This is a confident-error rate, not the fraction of incorrect answers:
    a wrong answer at low confidence does not count. See error_rate().
    """
    _check_lengths(correct, confidences, "confidences")
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
    _check_lengths(correct, confidences, "confidences")
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


def brier_score(correct: List[bool], confidences: List[float]) -> float:
    """Mean squared error between confidence and correctness (0 = perfect, 1 = worst)."""
    _check_lengths(correct, confidences, "confidences")
    if not correct:
        return 0.0
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((c - y) ** 2))


def risk_coverage_curve(correct: List[bool], confidences: List[float]) -> Tuple[List[Tuple[float, float]], float]:
    """
    Selective-prediction curve: answer the most confident examples first.

    Returns:
        (points, aurc) where points are (coverage, risk) pairs after each
        additional example is answered in descending-confidence order, and
        aurc is the area under that curve (lower is better).
    """
    _check_lengths(correct, confidences, "confidences")
    if not correct:
        return [], 0.0
    order = np.argsort(-np.asarray(confidences, dtype=float), kind='stable')
    wrong = 1.0 - np.asarray(correct, dtype=float)[order]
    n = len(wrong)
    risks = np.cumsum(wrong) / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / n
    points = list(zip(coverages.tolist(), risks.tolist()))
    return points, float(np.mean(risks))


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
    confidences: Optional[List[float]] = None,
    abstained: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """
    Summarize per-example correctness, confidence and abstention.

    Accuracy and error_rate are over all questions (an abstention is not a
    correct answer). coverage and answered_error_rate need abstention flags;
    confident_error_rate, calibration_error, brier and aurc need confidences.

    Returns:
        Dictionary of metric name -> score
    """
    metrics = {
        'accuracy': sum(correct) / len(correct) if correct else 0.0,
        'error_rate': error_rate(correct),
    }
    if abstained is not None:
        metrics['coverage'] = coverage(abstained)
        metrics['answered_error_rate'] = answered_error_rate(correct, abstained)
    if confidences is not None:
        metrics['confident_error_rate'] = confident_error_rate(correct, confidences)
        metrics['calibration_error'] = compute_calibration_error(correct, confidences)
        metrics['brier'] = brier_score(correct, confidences)
        metrics['aurc'] = risk_coverage_curve(correct, confidences)[1]
    return metrics
