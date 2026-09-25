"""
pipeline.py - Main pipeline orchestrator for Business Entity Resolution.

COMPLETE REDESIGN v2:
- Restartable: each stage caches its output; re-running skips completed stages
- Memory-efficient: chunked processing, garbage collection
- High recall blocking: 8 strategies including batched TF-IDF
- LightGBM model optimized for F0.5
- Full test inference with exactly the same normalization/blocking/features

Run as:
    python -m business_entity_resolution.src.pipeline
    python -m business_entity_resolution.src.pipeline --stage blocking
    python -m business_entity_resolution.src.pipeline --resume
"""

import argparse
import gc
import logging
import os
import pickle
import sys
import time
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add the project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from . import config as cfg
from .data_loader import (
    load_train_data,
    load_test_data,
    parse_ground_truth,
    print_dataset_stats,
)
from .normalization import normalize_dataframe
from .blocking import generate_candidates
from .features import build_feature_matrix, fit_tfidf_computers
from .labeling import build_pair_labels, check_blocking_recall, entity_level_train_val_split
from .train import train_model, predict_proba, save_model
from .evaluate import macro_f05, apply_threshold, print_evaluation_report
from .threshold import find_optimal_threshold, save_threshold
from .output_writer import (
    write_matching_results,
    write_candidate_pairs,
    validate_outputs_are_consistent,
)
from .predict import run_validator, _compute_proba_in_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _save_cache(obj, path: Path, label: str = "") -> None:
    """Save object to pickle cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    logger.info(f"Cached {label} → {path}")


def _load_cache(path: Path, label: str = ""):
    """Load object from pickle cache, or return None if missing."""
    if path.exists():
        logger.info(f"Loading cached {label} from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ─── Sub-step helpers ─────────────────────────────────────────────────────────

def _sample_training_pairs(
    pairs: List[Tuple[str, str]],
    labels: List[int],
    negative_ratio: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List, List]:
    """Undersample negatives to balance training data."""
    if negative_ratio is None:
        return pairs, labels

    rng = random.Random(seed)
    pos_idx  = [i for i, y in enumerate(labels) if y == 1]
    neg_idx  = [i for i, y in enumerate(labels) if y == 0]
    n_pos    = len(pos_idx)
    n_neg_keep = min(len(neg_idx), n_pos * negative_ratio)

    neg_sampled = rng.sample(neg_idx, n_neg_keep)
    keep_idx    = sorted(pos_idx + neg_sampled)

    pairs_out  = [pairs[i] for i in keep_idx]
    labels_out = [labels[i] for i in keep_idx]

    logger.info(
        f"Sampled training pairs: {len(pairs_out):,} "
        f"({n_pos:,} pos, {n_neg_keep:,} neg, ratio={negative_ratio})"
    )
    return pairs_out, labels_out


def _build_entity_dicts(s1, s2, s3):
    """Build entity_id → row dicts for fast lookup."""
    s1_dict = {}
    for row in s1.to_dict("records"):
        s1_dict[row["entity_id"]] = row

    target_dict = {}
    for row in s2.to_dict("records"):
        target_dict[row["entity_id"]] = row
    for row in s3.to_dict("records"):
        target_dict[row["entity_id"]] = row

    return s1_dict, target_dict


def _compute_features_in_chunks(
    pairs: List[Tuple[str, str]],
    s1_dict: Dict,
    target_dict: Dict,
    name_tfidf,
    addr_tfidf,
    chunk_size: int = 100_000,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute feature matrix in chunks. Returns (X, feature_names)."""
    if not pairs:
        return np.zeros((0, 0), dtype=np.float32), []

    all_chunks = []
    fn = feature_names

    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        chunk = pairs[start:end]
        feat_df, fn_chunk = build_feature_matrix(
            chunk, s1_dict, target_dict, name_tfidf, addr_tfidf
        )
        if fn is None:
            fn = fn_chunk
        # Align columns
        if fn:
            for col in fn:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[fn]
        all_chunks.append(feat_df.fillna(0.0).values.astype(np.float32))
        if (start // chunk_size) % 5 == 0:
            logger.info(f"  Features: {end:,}/{len(pairs):,}")
        del feat_df

    X = np.vstack(all_chunks)
    del all_chunks
    gc.collect()
    return X, fn or []


def _save_experiment(experiment: dict) -> None:
    """Append experiment record to experiments/results.csv."""
    path = cfg.EXPERIMENTS_CSV
    fieldnames = [
        "timestamp", "model", "blocking_strategy", "threshold",
        "precision", "recall", "f05", "candidate_count",
        "match_count", "singleton_count", "n_features", "notes",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(experiment)
    logger.info(f"Experiment saved to {path}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(resume: bool = False) -> None:
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  Business Entity Resolution Pipeline v2")
    logger.info(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # ── Stage 1: Load training data ───────────────────────────────────────────
    logger.info("\n[Stage 1] Loading training data...")
    s1, s2, s3, gt_df = load_train_data()
    gt_dict = parse_ground_truth(gt_df)
    print_dataset_stats(s1, s2, s3, gt_df, label="Training")

    # ── Stage 2: Normalize ────────────────────────────────────────────────────
    logger.info("\n[Stage 2] Normalizing training data...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    gc.collect()

    # ── Stage 3: Fit TF-IDF ──────────────────────────────────────────────────
    logger.info("\n[Stage 3] Fitting TF-IDF vectorizers...")
    tfidf_cache = _load_cache(cfg.TFIDF_CACHE, "TF-IDF") if resume else None
    if tfidf_cache is not None:
        name_tfidf, addr_tfidf = tfidf_cache
    else:
        sample_size = 300_000
        s1_names = s1["norm_name"].sample(min(sample_size, len(s1)), random_state=42).tolist()
        s2_names = s2["norm_name"].sample(min(sample_size, len(s2)), random_state=42).tolist()
        s3_names = s3["norm_name"].sample(min(sample_size, len(s3)), random_state=42).tolist()
        s1_addrs = s1["norm_address"].sample(min(sample_size, len(s1)), random_state=42).tolist()
        s2_addrs = s2["norm_address"].sample(min(sample_size, len(s2)), random_state=42).tolist()
        s3_addrs = s3["norm_address"].sample(min(sample_size, len(s3)), random_state=42).tolist()
        all_names = s1_names + s2_names + s3_names
        all_addrs = s1_addrs + s2_addrs + s3_addrs

        name_tfidf, addr_tfidf = fit_tfidf_computers(all_names, all_addrs)
        _save_cache((name_tfidf, addr_tfidf), cfg.TFIDF_CACHE, "TF-IDF")
        del s1_names, s2_names, s3_names, s1_addrs, s2_addrs, s3_addrs
        del all_names, all_addrs
        gc.collect()

    # ── Stage 4: Entity-level split ───────────────────────────────────────────
    logger.info("\n[Stage 4] Creating entity-level train/val split...")
    train_s1_ids, val_s1_ids = entity_level_train_val_split(
        gt_dict,
        val_fraction=cfg.TRAINING["validation_fraction"],
        random_seed=cfg.TRAINING["random_seed"],
    )
    train_s1_df = s1[s1["entity_id"].isin(train_s1_ids)].copy()
    val_s1_df   = s1[s1["entity_id"].isin(val_s1_ids)].copy()

    # ── Stage 5: Blocking on training data ────────────────────────────────────
    logger.info("\n[Stage 5] Generating training candidates (blocking)...")
    train_candidates = _load_cache(cfg.TRAIN_CANDIDATES_CACHE, "train candidates") if resume else None
    if train_candidates is None:
        logger.info(f"  Train S1 entities: {len(train_s1_ids):,}")
        train_candidates = generate_candidates(train_s1_df, s2, s3)
        _save_cache(train_candidates, cfg.TRAIN_CANDIDATES_CACHE, "train candidates")

    # Blocking recall check
    train_gt_subset = {k: v for k, v in gt_dict.items() if k in train_s1_ids}
    train_recall = check_blocking_recall(train_candidates, train_gt_subset, "train")

    # ── Stage 6: Build train feature matrix ───────────────────────────────────
    logger.info("\n[Stage 6] Building training feature matrix...")
    train_X_cache_path = cfg.CACHE_DIR / "train_X.npy"
    train_y_cache_path = cfg.CACHE_DIR / "train_y.npy"
    train_fn_cache_path = cfg.CACHE_DIR / "train_fn.pkl"
    train_pairs_cache   = cfg.CACHE_DIR / "train_pairs.pkl"

    entity_dicts_built = False
    if resume and train_X_cache_path.exists():
        logger.info("Loading cached training features...")
        X_train = np.load(str(train_X_cache_path))
        y_train = np.load(str(train_y_cache_path))
        with open(train_fn_cache_path, "rb") as f:
            feature_names = pickle.load(f)
        with open(train_pairs_cache, "rb") as f:
            train_pairs = pickle.load(f)
    else:
        train_pairs, train_labels = build_pair_labels(
            train_candidates, gt_dict, train_s1_ids
        )
        train_pairs, train_labels = _sample_training_pairs(
            train_pairs, train_labels,
            negative_ratio=cfg.TRAINING["negative_sample_ratio"],
            seed=cfg.TRAINING["random_seed"],
        )

        s1_dict, target_dict = _build_entity_dicts(s1, s2, s3)
        entity_dicts_built = True

        logger.info("Computing training features in chunks...")
        X_train, feature_names = _compute_features_in_chunks(
            train_pairs, s1_dict, target_dict, name_tfidf, addr_tfidf,
            chunk_size=cfg.TRAINING["feature_chunk_size"],
        )
        y_train = np.array(train_labels, dtype=np.float32)

        np.save(str(train_X_cache_path), X_train)
        np.save(str(train_y_cache_path), y_train)
        with open(train_fn_cache_path, "wb") as f:
            pickle.dump(feature_names, f)
        with open(train_pairs_cache, "wb") as f:
            pickle.dump(train_pairs, f)

    logger.info(
        f"Train matrix: {X_train.shape}, positives: {int(y_train.sum()):,}"
    )

    # ── Stage 7: Validation candidates + features ─────────────────────────────
    logger.info("\n[Stage 7] Generating validation candidates + features...")
    val_candidates = _load_cache(cfg.VAL_CANDIDATES_CACHE, "val candidates") if resume else None
    if val_candidates is None:
        val_candidates = generate_candidates(val_s1_df, s2, s3)
        _save_cache(val_candidates, cfg.VAL_CANDIDATES_CACHE, "val candidates")

    val_gt_subset = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
    check_blocking_recall(val_candidates, val_gt_subset, "val")

    val_pairs, val_labels = build_pair_labels(val_candidates, gt_dict, val_s1_ids)

    if not entity_dicts_built:
        s1_dict, target_dict = _build_entity_dicts(s1, s2, s3)

    X_val, _ = _compute_features_in_chunks(
        val_pairs, s1_dict, target_dict, name_tfidf, addr_tfidf,
        chunk_size=cfg.TRAINING["feature_chunk_size"],
        feature_names=feature_names,
    )
    y_val = np.array(val_labels, dtype=np.float32)

    # ── Stage 8: Train model ──────────────────────────────────────────────────
    logger.info("\n[Stage 8] Training model...")
    model = train_model(X_train, y_train, X_val, y_val, feature_names)

    # ── Stage 9: Validate ─────────────────────────────────────────────────────
    logger.info("\n[Stage 9] Evaluating on validation set...")
    val_proba = predict_proba(model, X_val)

    # ── Stage 10: Threshold optimization ─────────────────────────────────────
    logger.info("\n[Stage 10] Optimizing F0.5 threshold...")
    best_threshold, best_metrics = find_optimal_threshold(
        val_pairs, val_proba, val_gt_subset
    )

    # Build full predictions at best threshold
    val_predictions = apply_threshold(val_pairs, val_proba, best_threshold)
    for s1_id in val_s1_ids:
        if s1_id not in val_predictions:
            val_predictions[s1_id] = set()
    val_full_metrics = macro_f05(val_predictions, val_gt_subset)
    print_evaluation_report(val_full_metrics, label="Validation", threshold=best_threshold)

    save_threshold(best_threshold)

    # Clean up to free memory
    del X_train, X_val, train_candidates, val_candidates
    gc.collect()

    # ── Stage 11: Retrain on all training data ────────────────────────────────
    logger.info("\n[Stage 11] Retraining on all training data...")
    all_cands_cache = cfg.CACHE_DIR / "all_train_candidates.pkl"
    all_train_candidates = _load_cache(all_cands_cache, "all train candidates") if resume else None
    if all_train_candidates is None:
        logger.info("  Generating candidates for ALL training S1 entities...")
        all_train_candidates = generate_candidates(s1, s2, s3)
        _save_cache(all_train_candidates, all_cands_cache, "all train candidates")

    check_blocking_recall(all_train_candidates, gt_dict, "full_train")

    all_pairs, all_labels = build_pair_labels(all_train_candidates, gt_dict)
    all_pairs, all_labels = _sample_training_pairs(
        all_pairs, all_labels,
        negative_ratio=cfg.TRAINING["negative_sample_ratio"],
        seed=cfg.TRAINING["random_seed"],
    )

    logger.info("  Computing features for all training pairs...")
    if not entity_dicts_built:
        s1_dict, target_dict = _build_entity_dicts(s1, s2, s3)

    X_all, _ = _compute_features_in_chunks(
        all_pairs, s1_dict, target_dict, name_tfidf, addr_tfidf,
        chunk_size=cfg.TRAINING["feature_chunk_size"],
        feature_names=feature_names,
    )
    y_all = np.array(all_labels, dtype=np.float32)

    final_model = train_model(X_all, y_all, feature_names=feature_names)
    save_model(final_model, feature_names)

    del s1, s2, s3, X_all, all_train_candidates
    gc.collect()

    # ── Stage 12: Test prediction ─────────────────────────────────────────────
    logger.info("\n[Stage 12] Generating test predictions...")
    ts1, ts2, ts3 = load_test_data()
    all_test_s1_ids = ts1["entity_id"].tolist()
    logger.info(f"Test S1: {len(all_test_s1_ids):,}")

    ts1 = normalize_dataframe(ts1)
    ts2 = normalize_dataframe(ts2)
    ts3 = normalize_dataframe(ts3)

    # Update TF-IDF with test data too
    logger.info("Updating TF-IDF with test data...")
    t_names = (
        ts1["norm_name"].sample(min(100_000, len(ts1)), random_state=42).tolist()
        + ts2["norm_name"].sample(min(100_000, len(ts2)), random_state=42).tolist()
        + ts3["norm_name"].sample(min(100_000, len(ts3)), random_state=42).tolist()
    )
    t_addrs = (
        ts1["norm_address"].sample(min(100_000, len(ts1)), random_state=42).tolist()
        + ts2["norm_address"].sample(min(100_000, len(ts2)), random_state=42).tolist()
        + ts3["norm_address"].sample(min(100_000, len(ts3)), random_state=42).tolist()
    )
    name_tfidf.fit(t_names)
    addr_tfidf.fit(t_addrs)
    del t_names, t_addrs
    gc.collect()

    logger.info("Generating test candidates...")
    test_candidates = _load_cache(cfg.TEST_CANDIDATES_CACHE, "test candidates") if resume else None
    if test_candidates is None:
        test_candidates = generate_candidates(ts1, ts2, ts3)
        _save_cache(test_candidates, cfg.TEST_CANDIDATES_CACHE, "test candidates")

    test_s1_dict, test_target_dict = _build_entity_dicts(ts1, ts2, ts3)

    test_pairs = []
    for s1_id, cand_set in test_candidates.items():
        for cand_id in cand_set:
            test_pairs.append((s1_id, cand_id))
    logger.info(f"Total test candidate pairs: {len(test_pairs):,}")

    # ── Stage 13: Features + predict ──────────────────────────────────────────
    if test_pairs:
        logger.info("Computing test features and probabilities...")
        test_proba = _compute_proba_in_chunks(
            test_pairs, test_s1_dict, test_target_dict,
            final_model, feature_names, name_tfidf, addr_tfidf,
        )
        test_predictions = apply_threshold(test_pairs, test_proba, best_threshold)
    else:
        logger.warning("No test candidate pairs; all predictions will be singletons")
        test_predictions = {}

    # Ensure ALL S1 test entities are present
    for s1_id in all_test_s1_ids:
        if s1_id not in test_predictions:
            test_predictions[s1_id] = set()

    # ── Stage 14: Write output ─────────────────────────────────────────────────
    logger.info("\n[Stage 14] Writing output files...")

    # Write to BOTH output dirs (ber/output and student_resource/output)
    output_dirs = [cfg.OUTPUT_DIR, cfg.STUDENT_OUTPUT_DIR]
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_candidate_pairs(
            test_candidates, all_test_s1_ids,
            path=out_dir / "candidate_pairs.tsv"
        )
        write_matching_results(
            test_predictions, all_test_s1_ids,
            path=out_dir / "matching_results.tsv"
        )

    validate_outputs_are_consistent(test_predictions, test_candidates)

    # ── Stage 15: Run validator ────────────────────────────────────────────────
    logger.info("\n[Stage 15] Running submission validator...")
    validator_passed = run_validator()

    # ── Stage 16: Save experiment record ──────────────────────────────────────
    n_matched   = sum(1 for v in test_predictions.values() if v)
    n_singleton = sum(1 for v in test_predictions.values() if not v)
    total_match_pairs = sum(len(v) for v in test_predictions.values())

    experiment = {
        "timestamp":         datetime.now().isoformat(),
        "model":             "lightgbm",
        "blocking_strategy": "8_strategies_batched_tfidf",
        "threshold":         f"{best_threshold:.4f}",
        "precision":         f"{val_full_metrics['precision']:.4f}",
        "recall":            f"{val_full_metrics['recall']:.4f}",
        "f05":               f"{val_full_metrics['f05']:.4f}",
        "candidate_count":   len(test_pairs),
        "match_count":       total_match_pairs,
        "singleton_count":   n_singleton,
        "n_features":        len(feature_names),
        "notes":             f"validator_passed={validator_passed},train_recall={train_recall:.4f}",
    }
    _save_experiment(experiment)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    logger.info(f"  Validator: {'PASS' if validator_passed else 'FAIL'}")
    logger.info(f"  Validation F0.5: {val_full_metrics['f05']:.4f}")
    logger.info(f"  Validation Precision: {val_full_metrics['precision']:.4f}")
    logger.info(f"  Validation Recall: {val_full_metrics['recall']:.4f}")
    logger.info(f"  Threshold: {best_threshold:.4f}")
    logger.info(f"  Test matched: {n_matched:,} / {len(all_test_s1_ids):,} entities")
    logger.info(f"  Test total pairs: {total_match_pairs:,}")
    logger.info(f"  Output: {cfg.STUDENT_OUTPUT_DIR}")
    logger.info(f"{'='*70}")

    if not validator_passed:
        logger.error("Submission validator FAILED. Please fix errors before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached intermediate results")
    args = parser.parse_args()
    run_pipeline(resume=args.resume)
