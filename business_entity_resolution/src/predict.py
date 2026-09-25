"""
predict.py - Generate match predictions on the test set.

Steps:
1. Load test data
2. Normalize fields
3. Generate candidates (blocking)
4. Build feature matrix for candidate pairs
5. Predict match probabilities
6. Apply threshold
7. Write output files
"""

import logging
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import load_test_data, load_train_data
from .normalization import normalize_dataframe
from .blocking import generate_candidates
from .features import build_feature_matrix, fit_tfidf_computers
from .evaluate import apply_threshold
from .train import predict_proba, load_model, load_feature_names
from .threshold import load_threshold
from .output_writer import (
    write_matching_results,
    write_candidate_pairs,
    validate_outputs_are_consistent,
)

logger = logging.getLogger(__name__)


def predict_test(
    model=None,
    threshold: Optional[float] = None,
    name_tfidf=None,
    addr_tfidf=None,
    feature_names: Optional[List[str]] = None,
) -> None:
    """Run the full prediction pipeline on the test set.

    If model/threshold are not provided, they are loaded from disk.
    """
    logger.info("=" * 60)
    logger.info("  PREDICTION PHASE: Test Set")
    logger.info("=" * 60)

    # Load model if not provided
    if model is None:
        logger.info("Loading trained model from disk...")
        model = load_model()
        feature_names = load_feature_names()

    # Load threshold
    if threshold is None:
        threshold = load_threshold()
    logger.info(f"Using threshold: {threshold:.4f}")

    # Load test data
    logger.info("Loading test data...")
    s1, s2, s3 = load_test_data()
    all_test_s1_ids = s1["entity_id"].tolist()
    logger.info(f"Test S1 entities: {len(all_test_s1_ids):,}")
    logger.info(f"Test S2 records:  {len(s2):,}")
    logger.info(f"Test S3 records:  {len(s3):,}")

    # Normalize
    logger.info("Normalizing test data...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)

    # Fit TF-IDF if not provided
    if name_tfidf is None or addr_tfidf is None:
        logger.info("Fitting TF-IDF on test data...")
        # Load train data too for fitting TF-IDF
        try:
            tr_s1, tr_s2, tr_s3, _ = load_train_data()
            tr_s1 = normalize_dataframe(tr_s1)
            tr_s2 = normalize_dataframe(tr_s2)
            tr_s3 = normalize_dataframe(tr_s3)
            all_names = (
                tr_s1["norm_name"].tolist() + tr_s2["norm_name"].tolist() +
                tr_s3["norm_name"].tolist() + s1["norm_name"].tolist() +
                s2["norm_name"].tolist() + s3["norm_name"].tolist()
            )
            all_addrs = (
                tr_s1["norm_address"].tolist() + tr_s2["norm_address"].tolist() +
                tr_s3["norm_address"].tolist() + s1["norm_address"].tolist() +
                s2["norm_address"].tolist() + s3["norm_address"].tolist()
            )
        except Exception:
            all_names = s1["norm_name"].tolist() + s2["norm_name"].tolist() + s3["norm_name"].tolist()
            all_addrs = s1["norm_address"].tolist() + s2["norm_address"].tolist() + s3["norm_address"].tolist()

        from .features import fit_tfidf_computers
        name_tfidf, addr_tfidf = fit_tfidf_computers(all_names, all_addrs)

    # Generate candidates
    logger.info("Generating test candidates (blocking)...")
    candidates = generate_candidates(s1, s2, s3)

    # Build entity lookup dicts
    s1_dict     = {row["entity_id"]: row for _, row in s1.iterrows()}
    target_dict = {}
    for _, row in s2.iterrows():
        target_dict[row["entity_id"]] = row
    for _, row in s3.iterrows():
        target_dict[row["entity_id"]] = row

    # Build pair list
    pairs = []
    for s1_id, cand_set in candidates.items():
        for cand_id in cand_set:
            pairs.append((s1_id, cand_id))

    logger.info(f"Total test candidate pairs: {len(pairs):,}")

    if not pairs:
        logger.warning("No candidate pairs found! Writing empty predictions.")
        predictions = {s1_id: set() for s1_id in all_test_s1_ids}
    else:
        # Feature computation in chunks (memory-safe)
        logger.info("Computing features for test pairs...")
        all_proba = _compute_proba_in_chunks(
            pairs, s1_dict, target_dict,
            model, feature_names, name_tfidf, addr_tfidf
        )

        # Apply threshold
        logger.info(f"Applying threshold {threshold:.4f}...")
        predictions = apply_threshold(pairs, all_proba, threshold)

    # Ensure all S1 entities are in predictions
    for s1_id in all_test_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    # Write outputs
    logger.info("Writing output files...")
    write_candidate_pairs(candidates, all_test_s1_ids)
    write_matching_results(predictions, all_test_s1_ids)

    # Consistency check
    validate_outputs_are_consistent(predictions, candidates)

    # Summary
    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    logger.info(f"\nTest prediction summary:")
    logger.info(f"  Total S1 entities:  {len(all_test_s1_ids):,}")
    logger.info(f"  With matches:       {n_matched:,}")
    logger.info(f"  Singletons:         {n_singleton:,}")
    total_preds = sum(len(v) for v in predictions.values())
    logger.info(f"  Total match pairs:  {total_preds:,}")


def _compute_proba_in_chunks(
    pairs, s1_dict, target_dict, model, feature_names, name_tfidf, addr_tfidf,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Compute match probabilities for pairs in memory-safe chunks."""
    all_proba = np.zeros(len(pairs), dtype=np.float32)

    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        chunk = pairs[start:end]

        feat_df, _ = build_feature_matrix(
            chunk, s1_dict, target_dict, name_tfidf, addr_tfidf
        )

        if feat_df.empty:
            all_proba[start:end] = 0.0
            continue

        # Ensure features are in the same order as training
        if feature_names:
            for col in feature_names:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[feature_names]

        X = feat_df.values.astype(np.float32)
        proba = predict_proba(model, X)
        all_proba[start:end] = proba

        logger.info(f"  Processed pairs {start:,}-{end:,} / {len(pairs):,}")

    return all_proba


def run_validator() -> bool:
    """Run the official submission validator script."""
    logger.info("\nRunning submission validator...")
    validator = cfg.VALIDATE_SCRIPT
    matching  = cfg.OUTPUT_MATCHING
    candidate = cfg.OUTPUT_CANDIDATES
    test_dir  = cfg.TEST_DIR

    cmd = [
        sys.executable, str(validator),
        "--matching",   str(matching),
        "--candidate",  str(candidate),
        "--test-dir",   str(test_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("VALIDATOR ERRORS:")
        print(result.stderr)
        return False
    return True
