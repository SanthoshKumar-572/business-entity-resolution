"""
labeling.py - Construct pair-level training labels from ground truth.

For each candidate pair (S1_id, target_id):
  label = 1 if target_id appears in ground_truth[S1_id]
  label = 0 otherwise

Careful handling of singletons (S1 entities with no ground-truth matches).
"""

import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

from . import config as cfg
from .data_loader import parse_ground_truth

logger = logging.getLogger(__name__)


def build_pair_labels(
    candidates: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    s1_ids_in_split: Set[str] = None,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Convert candidate pairs + ground truth into (pair, label) lists.

    Args:
        candidates: dict mapping S1_id → set of candidate target IDs
        gt_dict:    dict mapping S1_id → set of true matching target IDs
        s1_ids_in_split: if provided, only include pairs for these S1 IDs

    Returns:
        pairs:  list of (s1_id, target_id) tuples
        labels: list of 0/1 labels
    """
    pairs  = []
    labels = []

    for s1_id, cand_set in candidates.items():
        if s1_ids_in_split is not None and s1_id not in s1_ids_in_split:
            continue

        true_matches = gt_dict.get(s1_id, set())

        for cand_id in cand_set:
            label = 1 if cand_id in true_matches else 0
            pairs.append((s1_id, cand_id))
            labels.append(label)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_rate = 100.0 * n_pos / len(labels) if labels else 0.0
    logger.info(
        f"Built {len(labels):,} pairs: "
        f"{n_pos:,} positive ({pos_rate:.2f}%), "
        f"{n_neg:,} negative"
    )
    return pairs, labels


def check_blocking_recall(
    candidates: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    label: str = "blocking",
) -> float:
    """Compute blocking recall: fraction of true positives in candidates.

    This is the upper bound of recall for the ML matching model.
    """
    total_true = 0
    captured   = 0
    missed_examples = []

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue  # skip singletons

        cand_set = candidates.get(s1_id, set())
        for tm in true_matches:
            total_true += 1
            if tm in cand_set:
                captured += 1
            else:
                missed_examples.append((s1_id, tm))

    recall = captured / total_true if total_true > 0 else 0.0
    logger.info(
        f"[{label}] Blocking recall: {captured}/{total_true} = {recall:.4f} "
        f"({len(missed_examples)} true matches NOT in candidates)"
    )
    if missed_examples[:5]:
        logger.info(f"  Examples of missed matches: {missed_examples[:5]}")

    return recall


def entity_level_train_val_split(
    gt_dict: Dict[str, Set[str]],
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Split S1 entity IDs into train and validation sets.

    Split is at the entity level to prevent leakage:
    all pairs for a given S1 entity go into the same split.

    Returns:
        (train_s1_ids, val_s1_ids)
    """
    import random
    rng = random.Random(random_seed)

    all_s1_ids = list(gt_dict.keys())
    rng.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * val_fraction)
    val_ids   = set(all_s1_ids[:n_val])
    train_ids = set(all_s1_ids[n_val:])

    logger.info(
        f"Entity-level split: "
        f"{len(train_ids):,} train entities, "
        f"{len(val_ids):,} val entities"
    )
    return train_ids, val_ids
