"""
smoke_test.py - Quick end-to-end smoke test on a small sample.

Tests the full pipeline on a tiny slice of the training data to ensure
all stages work correctly before running the full pipeline.

STAGE 1: 500 S1 entities, 10K S2/S3 each
STAGE 2: 2000 S1 entities, 50K S2/S3 each
STAGE 3: Run on larger samples if needed

Run from student_resource/:
    python -m business_entity_resolution.smoke_test
    python -m business_entity_resolution.smoke_test --stage 2
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_smoke_test(n_s1=500, n_s2=10000, n_s3=10000):
    """Run a minimal pipeline on a tiny sample.

    Loads S2/S3 first, then finds S1 entities whose GT matches appear in
    that S2/S3 sample, so blocking can actually find real matches.
    """
    from business_entity_resolution.src import config as cfg
    from business_entity_resolution.src.data_loader import parse_ground_truth
    from business_entity_resolution.src.normalization import normalize_dataframe
    from business_entity_resolution.src.blocking import generate_candidates
    from business_entity_resolution.src.features import build_feature_matrix, fit_tfidf_computers
    from business_entity_resolution.src.labeling import (
        build_pair_labels, check_blocking_recall, entity_level_train_val_split
    )
    from business_entity_resolution.src.train import train_model, predict_proba
    from business_entity_resolution.src.evaluate import (
        macro_f05, apply_threshold, print_evaluation_report
    )
    from business_entity_resolution.src.threshold import find_optimal_threshold

    import time
    import gc
    t0 = time.time()

    logger.info("=" * 60)
    logger.info(f"  Smoke Test: S1={n_s1}, S2={n_s2}, S3={n_s3}")
    logger.info("=" * 60)

    # Load S2/S3 first to know what target IDs exist
    logger.info("Loading S2/S3 samples...")
    s2 = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", dtype=str, nrows=n_s2).fillna("")
    s3 = pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", dtype=str, nrows=n_s3).fillna("")
    target_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    logger.info(f"Target IDs available: {len(target_ids):,}")

    # Load GT and find S1 entities whose matches appear in our S2/S3 sample
    logger.info("Scanning GT for S1 entities with matches in sample...")
    gt_scan_rows = min(50000, 5 * n_s1)
    gt_df = pd.read_csv(cfg.TRAIN_GT, sep="\t", dtype=str, nrows=gt_scan_rows).fillna("")
    gt_dict_scan = parse_ground_truth(gt_df)

    valid_s1_ids = []
    for s1_id, matches in gt_dict_scan.items():
        # Include entities with matches in sample AND singletons
        if not matches or bool(matches & target_ids):
            valid_s1_ids.append(s1_id)
        if len(valid_s1_ids) >= n_s1:
            break

    logger.info(f"Selected {len(valid_s1_ids)} S1 entities for smoke test")
    if len(valid_s1_ids) < 10:
        logger.error("Not enough valid S1 entities — increase n_s2/n_s3 or n_s1!")
        return None

    # Load S1 source file and filter to valid IDs
    logger.info("Loading S1 records...")
    s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", dtype=str).fillna("")
    s1 = s1_full[s1_full["entity_id"].isin(set(valid_s1_ids))].reset_index(drop=True)
    gt_dict = {k: v for k, v in gt_dict_scan.items() if k in set(valid_s1_ids)}
    del s1_full

    n_with_matches = sum(1 for v in gt_dict.values() if v & target_ids)
    n_singletons   = sum(1 for v in gt_dict.values() if not v)
    logger.info(f"S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT={len(gt_dict)}")
    logger.info(f"S1 with matches in sample: {n_with_matches}, singletons: {n_singletons}")

    # Normalize
    logger.info("Normalizing...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)

    # Fit TF-IDF
    all_names = s1["norm_name"].tolist() + s2["norm_name"].tolist() + s3["norm_name"].tolist()
    all_addrs = s1["norm_address"].tolist() + s2["norm_address"].tolist() + s3["norm_address"].tolist()
    name_tfidf, addr_tfidf = fit_tfidf_computers(all_names, all_addrs)
    del all_names, all_addrs
    gc.collect()

    # Split
    train_ids, val_ids = entity_level_train_val_split(gt_dict, val_fraction=0.2)
    train_s1 = s1[s1["entity_id"].isin(train_ids)].copy()
    val_s1   = s1[s1["entity_id"].isin(val_ids)].copy()

    # Blocking
    logger.info("Generating train candidates...")
    train_cands = generate_candidates(train_s1, s2, s3)
    train_gt_sub = {k: v for k, v in gt_dict.items() if k in train_ids}
    train_recall = check_blocking_recall(train_cands, train_gt_sub)

    logger.info("Generating val candidates...")
    val_cands = generate_candidates(val_s1, s2, s3)
    val_gt_sub = {k: v for k, v in gt_dict.items() if k in val_ids}
    val_recall = check_blocking_recall(val_cands, val_gt_sub)

    # Candidate stats
    all_cands = {**train_cands, **val_cands}
    cand_sizes = [len(v) for v in all_cands.values()]
    import statistics
    if cand_sizes:
        logger.info(f"Candidate stats: avg={statistics.mean(cand_sizes):.1f}, "
                    f"median={statistics.median(cand_sizes):.0f}, "
                    f"p95={sorted(cand_sizes)[int(0.95*len(cand_sizes))]}, "
                    f"max={max(cand_sizes)}")

    # Build labels
    train_pairs, train_labels = build_pair_labels(train_cands, gt_dict, train_ids)
    val_pairs, val_labels     = build_pair_labels(val_cands, gt_dict, val_ids)
    logger.info(
        f"Train pairs: {len(train_pairs):,} ({sum(train_labels)} pos) | "
        f"Val pairs: {len(val_pairs):,} ({sum(val_labels)} pos)"
    )

    if sum(train_labels) == 0:
        logger.warning("No positive training pairs found! Skipping model training.")
        return {"f05": 0.0, "precision": 0.0, "recall": 0.0,
                "blocking_recall": train_recall}

    # Build entity dicts
    s1_dict = {row["entity_id"]: row for row in s1.to_dict("records")}
    target_dict = {}
    for row in s2.to_dict("records"):
        target_dict[row["entity_id"]] = row
    for row in s3.to_dict("records"):
        target_dict[row["entity_id"]] = row

    # Build features
    logger.info("Computing features...")
    train_feat_df, feature_names = build_feature_matrix(
        train_pairs, s1_dict, target_dict, name_tfidf, addr_tfidf
    )
    val_feat_df, _ = build_feature_matrix(
        val_pairs, s1_dict, target_dict, name_tfidf, addr_tfidf
    )
    for col in feature_names:
        if col not in val_feat_df.columns:
            val_feat_df[col] = 0.0
    if feature_names:
        val_feat_df = val_feat_df[feature_names]

    X_train = train_feat_df.fillna(0.0).values.astype(np.float32)
    y_train = np.array(train_labels, dtype=np.float32)
    X_val = (
        val_feat_df.fillna(0.0).values.astype(np.float32)
        if len(val_pairs) > 0
        else np.zeros((0, X_train.shape[1]), dtype=np.float32)
    )
    y_val = np.array(val_labels, dtype=np.float32)

    logger.info(f"Feature matrix: {X_train.shape}, {y_train.sum():.0f} positives")

    # Train
    logger.info("Training model...")
    model = train_model(
        X_train, y_train,
        X_val if len(X_val) > 0 else None,
        y_val if len(y_val) > 0 else None,
        feature_names,
    )

    # Threshold optimization
    val_proba = predict_proba(model, X_val) if len(X_val) > 0 else np.array([])
    if len(val_proba) > 0 and sum(val_labels) > 0:
        logger.info("Optimizing threshold...")
        best_threshold, best_metrics = find_optimal_threshold(
            val_pairs, val_proba, val_gt_sub
        )
    else:
        best_threshold = 0.70
        logger.warning("No val positives; using default threshold")

    # Evaluate
    if len(val_proba) > 0:
        val_preds = apply_threshold(val_pairs, val_proba, best_threshold)
    else:
        val_preds = {}
    for s1_id in val_ids:
        if s1_id not in val_preds:
            val_preds[s1_id] = set()
    final_metrics = macro_f05(val_preds, val_gt_sub)
    print_evaluation_report(final_metrics, "Smoke Test Validation", threshold=best_threshold)

    # Test output writing
    from business_entity_resolution.src.output_writer import (
        write_matching_results, write_candidate_pairs
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        m_path = Path(tmpdir) / "matching.tsv"
        c_path = Path(tmpdir) / "candidates.tsv"
        val_id_list = list(val_ids)
        write_matching_results(val_preds, val_id_list, path=m_path)
        write_candidate_pairs(val_cands, val_id_list, path=c_path)
        df_m = pd.read_csv(m_path, sep="\t", dtype=str)
        df_c = pd.read_csv(c_path, sep="\t", dtype=str)
        assert len(df_m) == len(val_id_list), (
            f"matching: expected {len(val_id_list)}, got {len(df_m)}"
        )
        assert len(df_c) == len(val_id_list), (
            f"candidates: expected {len(val_id_list)}, got {len(df_c)}"
        )
        logger.info(f"Output validation: PASSED ({len(df_m)} rows each)")

    elapsed = time.time() - t0
    logger.info("\n=== SMOKE TEST PASSED ===")
    logger.info(f"  Train recall (blocking): {train_recall:.4f}")
    logger.info(f"  Val recall (blocking):   {val_recall:.4f}")
    logger.info(f"  Best threshold:          {best_threshold:.4f}")
    logger.info(f"  Validation F0.5:         {final_metrics['f05']:.4f}")
    logger.info(f"  Validation Precision:    {final_metrics['precision']:.4f}")
    logger.info(f"  Validation Recall:       {final_metrics['recall']:.4f}")
    logger.info(f"  Feature count:           {len(feature_names)}")
    logger.info(f"  Runtime:                 {elapsed/60:.1f} minutes")

    final_metrics["blocking_recall_train"] = train_recall
    final_metrics["blocking_recall_val"] = val_recall
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-s1", type=int, default=500)
    parser.add_argument("--n-s2", type=int, default=10000)
    parser.add_argument("--n-s3", type=int, default=10000)
    args = parser.parse_args()
    metrics = run_smoke_test(n_s1=args.n_s1, n_s2=args.n_s2, n_s3=args.n_s3)
