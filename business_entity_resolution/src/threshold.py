"""
threshold.py - F0.5-optimized threshold selection.

Searches over a grid of thresholds to find the one that maximizes F0.5
on the validation set. Uses macro-averaged F0.5 matching competition metric.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from . import config as cfg
from .evaluate import macro_f05, apply_threshold

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    val_pairs: List[Tuple[str, str]],
    val_proba: np.ndarray,
    val_gt: Dict[str, Set[str]],
    thresholds: List[float] = None,
    beta: float = 0.5,
) -> Tuple[float, Dict]:
    """Search for the threshold that maximizes F0.5 on the validation set.

    Args:
        val_pairs: list of (s1_id, target_id)
        val_proba: array of predicted probabilities
        val_gt:    dict mapping S1_id → set of true match IDs (validation only)
        thresholds: list of thresholds to try
        beta:      F-beta parameter (0.5 = precision-heavy)

    Returns:
        (best_threshold, results_dict)
    """
    if thresholds is None:
        thresholds = cfg.THRESHOLD["search_thresholds"]

    results = []
    for threshold in thresholds:
        preds = apply_threshold(val_pairs, val_proba, threshold)
        # Include all val GT entities, even those with no candidates
        for s1_id in val_gt:
            if s1_id not in preds:
                preds[s1_id] = set()
        metrics = macro_f05(preds, val_gt)
        results.append({
            "threshold": threshold,
            "f05":       metrics["f05"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "n_predicted": metrics["n_predicted_matches"],
            "n_correct_singletons": metrics["n_correct_singletons"],
        })
        logger.info(
            f"  threshold={threshold:.3f} | "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F0.5={metrics['f05']:.4f} | "
            f"matches={metrics['n_predicted_matches']:,}"
        )

    # Find best F0.5
    best = max(results, key=lambda x: x["f05"])
    best_threshold = best["threshold"]

    # Fine-grained search around the best threshold
    best_threshold, best, results = _fine_search(
        val_pairs, val_proba, val_gt, best_threshold, results
    )

    logger.info(
        f"\nBest threshold: {best_threshold:.4f} "
        f"→ F0.5={best['f05']:.4f} "
        f"P={best['precision']:.4f} R={best['recall']:.4f}"
    )

    return best_threshold, best


def _fine_search(
    val_pairs, val_proba, val_gt, coarse_best, coarse_results
) -> Tuple[float, Dict, List]:
    """Fine-grained search in ±0.05 around the coarse best threshold."""
    lo = max(0.01, coarse_best - 0.05)
    hi = min(0.99, coarse_best + 0.05)
    fine_thresholds = np.linspace(lo, hi, 21).tolist()

    for threshold in fine_thresholds:
        # Skip if already in coarse results
        if any(abs(r["threshold"] - threshold) < 1e-4 for r in coarse_results):
            continue
        preds = apply_threshold(val_pairs, val_proba, threshold)
        for s1_id in val_gt:
            if s1_id not in preds:
                preds[s1_id] = set()
        metrics = macro_f05(preds, val_gt)
        coarse_results.append({
            "threshold": threshold,
            "f05":       metrics["f05"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "n_predicted": metrics["n_predicted_matches"],
            "n_correct_singletons": metrics["n_correct_singletons"],
        })

    best = max(coarse_results, key=lambda x: x["f05"])
    return best["threshold"], best, coarse_results


def save_threshold(threshold: float, path=None) -> None:
    """Save the selected threshold to disk."""
    path = path or cfg.THRESHOLD_PATH
    with open(path, "w") as f:
        f.write(str(threshold))
    logger.info(f"Threshold {threshold:.4f} saved to {path}")


def load_threshold(path=None) -> float:
    """Load threshold from disk, or return default."""
    path = path or cfg.THRESHOLD_PATH
    try:
        with open(path) as f:
            t = float(f.read().strip())
        logger.info(f"Loaded threshold: {t:.4f}")
        return t
    except (FileNotFoundError, ValueError):
        t = cfg.THRESHOLD["default_threshold"]
        logger.warning(f"Could not load threshold from {path}, using default: {t}")
        return t
