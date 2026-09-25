"""
evaluate.py - F0.5 evaluation metric and validation reporting.

Implements the competition F0.5 metric:
  F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

Computed as macro-average across Source 1 entities.
"""

import logging
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Per-entity F0.5 ─────────────────────────────────────────────────────────

def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score."""
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def entity_f05(
    predicted: Set[str],
    ground_truth: Set[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F0.5 for a single S1 entity.

    Competition rules:
    - If both predicted and ground_truth are empty → F0.5 = 1.0 (correct singleton)
    - If predicted is non-empty but ground_truth is empty → F0.5 = 0.0 (false merge)
    - If predicted is empty but ground_truth is non-empty → F0.5 = 0.0 (missed match)
    - Otherwise → normal computation
    """
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0  # correct singleton prediction

    if not predicted:
        return 0.0, 0.0, 0.0  # missed all matches

    if not ground_truth:
        return 0.0, 0.0, 0.0  # false positives on a singleton

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall    = tp / len(ground_truth)
    f05       = fbeta_score(precision, recall, beta=0.5)
    return precision, recall, f05


# ─── Macro F0.5 across entities ───────────────────────────────────────────────

def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute macro-averaged F0.5 across all S1 entities.

    Args:
        predictions:  dict mapping S1_id → set of predicted match IDs
        ground_truth: dict mapping S1_id → set of true match IDs

    Returns:
        dict with keys: precision, recall, f05, n_entities, n_correct_singletons,
                        n_false_positives, n_false_negatives, n_predicted_matches
    """
    precisions, recalls, f05s = [], [], []
    n_correct_singletons = 0
    n_false_positive_singletons = 0
    n_false_negative_singletons = 0
    n_predicted_matches = 0

    # Evaluate every entity that appears in ground truth
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f = entity_f05(pred_set, true_set)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f)
        n_predicted_matches += len(pred_set)

        if not true_set and not pred_set:
            n_correct_singletons += 1
        elif not true_set and pred_set:
            n_false_positive_singletons += 1
        elif true_set and not pred_set:
            n_false_negative_singletons += 1

    return {
        "precision":                    float(np.mean(precisions)) if precisions else 0.0,
        "recall":                       float(np.mean(recalls))    if recalls    else 0.0,
        "f05":                          float(np.mean(f05s))       if f05s       else 0.0,
        "n_entities":                   len(ground_truth),
        "n_correct_singletons":         n_correct_singletons,
        "n_false_positive_singletons":  n_false_positive_singletons,
        "n_false_negative_singletons":  n_false_negative_singletons,
        "n_predicted_matches":          n_predicted_matches,
    }


# ─── Pair-level evaluation (for model calibration) ───────────────────────────

def pair_level_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute precision, recall, F0.5 at the pair level."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05       = fbeta_score(precision, recall, beta=0.5)

    return {"precision": precision, "recall": recall, "f05": f05, "tp": tp, "fp": fp, "fn": fn}


# ─── Threshold → predictions ──────────────────────────────────────────────────

def apply_threshold(
    pairs: List[Tuple[str, str]],
    proba: np.ndarray,
    threshold: float,
) -> Dict[str, Set[str]]:
    """Convert probabilities to entity-level predictions using a threshold.

    Args:
        pairs:     list of (s1_id, target_id)
        proba:     array of match probabilities
        threshold: float

    Returns:
        dict mapping S1_id → set of predicted match IDs
    """
    predictions: Dict[str, Set[str]] = {}
    for (s1_id, tgt_id), prob in zip(pairs, proba):
        if s1_id not in predictions:
            predictions[s1_id] = set()
        if prob >= threshold:
            predictions[s1_id].add(tgt_id)
    return predictions


# ─── Full validation report ───────────────────────────────────────────────────

def print_evaluation_report(
    metrics: Dict[str, float],
    label: str = "Validation",
    threshold: Optional[float] = None,
) -> None:
    """Print a formatted evaluation report."""
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  {label} Results")
    if threshold is not None:
        print(f"  Threshold: {threshold:.4f}")
    print(sep)
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F0.5 (macro):        {metrics['f05']:.4f}")
    print(f"  Entities evaluated:  {metrics['n_entities']:,}")
    print(f"  Predicted matches:   {metrics['n_predicted_matches']:,}")
    print(f"  Correct singletons:  {metrics.get('n_correct_singletons', 0):,}")
    print(f"  FP singletons:       {metrics.get('n_false_positive_singletons', 0):,}")
    print(f"  FN singletons:       {metrics.get('n_false_negative_singletons', 0):,}")
    print(sep)

