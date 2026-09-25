"""
output_writer.py - Generate competition-ready output TSV files.

Produces:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv

Both files must:
  - Be TAB-separated (UTF-8)
  - Have exactly one row per Source 1 test entity
  - Use comma-separated ID lists (no spaces)
  - Contain only S2-/S3- IDs (no S1- self-matches, no duplicates)
"""

import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


def _ids_to_str(id_set: Set[str]) -> str:
    """Convert a set of IDs to a sorted comma-separated string."""
    if not id_set:
        return ""
    return ",".join(sorted(id_set))


def write_matching_results(
    predictions: Dict[str, Set[str]],
    all_s1_ids: list,
    path=None,
) -> None:
    """Write matching_results.tsv.

    Args:
        predictions: dict mapping S1_id → set of predicted matching IDs
        all_s1_ids:  all S1 entity IDs that must appear in output
        path:        output file path (defaults to config)
    """
    path = Path(path or cfg.OUTPUT_MATCHING)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        matched = predictions.get(s1_id, set())
        # Validate: only S2-/S3- IDs
        matched = {
            eid for eid in matched
            if eid.startswith("S2-") or eid.startswith("S3-")
        }
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": _ids_to_str(matched),
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Wrote {len(df):,} rows to {path}")

    n_matches   = sum(1 for r in rows if r["matched_entity_ids"])
    n_singletons = sum(1 for r in rows if not r["matched_entity_ids"])
    logger.info(f"  Matched entities: {n_matches:,}")
    logger.info(f"  Singleton entities: {n_singletons:,}")


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    all_s1_ids: list,
    path=None,
) -> None:
    """Write candidate_pairs.tsv.

    Args:
        candidates: dict mapping S1_id → set of candidate IDs (from blocking)
        all_s1_ids: all S1 entity IDs that must appear in output
        path:       output file path (defaults to config)
    """
    path = Path(path or cfg.OUTPUT_CANDIDATES)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        cand_set = candidates.get(s1_id, set())
        # Validate: only S2-/S3- IDs
        cand_set = {
            eid for eid in cand_set
            if eid.startswith("S2-") or eid.startswith("S3-")
        }
        rows.append({
            "source1_entity_id":   s1_id,
            "candidate_entity_ids": _ids_to_str(cand_set),
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Wrote {len(df):,} rows to {path}")

    n_with_candidates = sum(1 for r in rows if r["candidate_entity_ids"])
    total_candidates  = sum(len(candidates.get(s1_id, set())) for s1_id in all_s1_ids)
    logger.info(f"  S1 with candidates: {n_with_candidates:,}")
    logger.info(f"  Total candidate pairs: {total_candidates:,}")


def validate_outputs_are_consistent(
    predictions: Dict[str, Set[str]],
    candidates:  Dict[str, Set[str]],
) -> bool:
    """Check that all predicted matches appear in candidates (required by validator)."""
    issues = []
    for s1_id, pred_set in predictions.items():
        cand_set = candidates.get(s1_id, set())
        extra = pred_set - cand_set
        if extra:
            issues.append(f"{s1_id}: matches not in candidates: {extra}")

    if issues:
        logger.warning(f"Found {len(issues)} consistency issues (matches not in candidates):")
        for issue in issues[:5]:
            logger.warning(f"  {issue}")
        return False
    else:
        logger.info("Output consistency check: PASSED (all matches are in candidates)")
        return True
