"""
06_final_inference.py
======================
Apply the trained LightGBM model to test features and generate
output/matching_results.tsv.

Pipeline:
  1. Load model from model/lgbm_model.txt
  2. Load threshold from model/threshold.txt
  3. Load test features from output/features_test.parquet
  4. Predict match probabilities
  5. Apply threshold to select matches
  6. Load ALL test S1 IDs (ensure every ID gets a row, even with no match)
  7. Write output/matching_results.tsv

Output format:
    source1_entity_id   matched_entity_ids

  - One row per test S1 entity
  - matched_entity_ids: comma-separated S2/S3 IDs (empty string for no match)
  - Every matched ID MUST exist in test_candidate_pairs.tsv
"""

import os
import sys
import gc
import csv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "model")
TEST_DIR = os.path.join(ROOT, "dataset", "test")

MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_model.txt")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "threshold.txt")

TEST_FEATURES = os.path.join(OUTPUT_DIR, "features_test.parquet")
TEST_CANDIDATES = os.path.join(OUTPUT_DIR, "test_candidate_pairs.tsv")
TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")

BATCH_SIZE = 100_000


def load_test_s1_ids() -> set:
    """Load all test S1 entity IDs."""
    ids = set()
    reader = pd.read_csv(TEST_S1, sep="\t", dtype=str, chunksize=100_000, usecols=["entity_id"])
    for chunk in reader:
        ids.update(chunk["entity_id"].dropna().tolist())
        del chunk
    print(f"  Test S1 entities: {len(ids):,}")
    return ids


def load_test_candidates() -> dict:
    """
    Load test_candidate_pairs.tsv as dict: s1_id → set(candidate_ids).
    Used to verify that every matched ID is in the candidate set.
    """
    print(f"  Loading test candidates from {TEST_CANDIDATES} ...")
    cand_map = {}
    if not os.path.isfile(TEST_CANDIDATES):
        print(f"  WARNING: {TEST_CANDIDATES} not found.")
        return cand_map

    reader = pd.read_csv(
        TEST_CANDIDATES, sep="\t", dtype=str, chunksize=10_000,
    )
    for chunk in reader:
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            s1_id = row.source1_entity_id
            cand_str = (row.candidate_entity_ids if hasattr(row, 'candidate_entity_ids') else "").strip()
            cand_map[s1_id] = set(cand_str.split(",")) if cand_str else set()
        del chunk
    print(f"  Loaded {len(cand_map):,} S1 candidate entries")
    return cand_map


def main():
    print("=" * 60)
    print("Final Inference — Generating matching_results.tsv")
    print("=" * 60)

    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed.")
        sys.exit(1)

    # Load model
    print(f"\n[Model] Loading {MODEL_FILE} ...")
    if not os.path.isfile(MODEL_FILE):
        print(f"  ERROR: {MODEL_FILE} not found. Run 04_train_model.py first.")
        sys.exit(1)
    model = lgb.Booster(model_file=MODEL_FILE)
    print(f"  Model loaded. Best iteration: {model.best_iteration}")

    # Load threshold
    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())
    print(f"  Using threshold: {threshold}")

    # Load test features
    print(f"\n[Features] Loading {TEST_FEATURES} ...")
    if not os.path.isfile(TEST_FEATURES):
        print(f"  ERROR: {TEST_FEATURES} not found. Run 03_feature_engineering.py test first.")
        sys.exit(1)
    df = pd.read_parquet(TEST_FEATURES)
    print(f"  Loaded {len(df):,} test pairs")

    META_COLS = {"source1_entity_id", "candidate_entity_id", "label"}
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Feature cols: {len(feature_cols)}")

    # Batch prediction
    print(f"\n[Predict] Running model inference (batch_size={BATCH_SIZE:,}) ...")
    all_probs = []
    X = df[feature_cols].values.astype(np.float32)

    for start in tqdm(range(0, len(X), BATCH_SIZE), desc="  Batches"):
        batch = X[start:start + BATCH_SIZE]
        probs = model.predict(batch, num_iteration=model.best_iteration)
        all_probs.append(probs)

    df["prob"] = np.concatenate(all_probs)
    del X, all_probs
    gc.collect()

    # Apply threshold
    print(f"\n[Threshold] Applying threshold {threshold} ...")
    df["match"] = (df["prob"] >= threshold).astype(int)
    n_matches = df["match"].sum()
    print(f"  Matched pairs: {n_matches:,} / {len(df):,}")

    # Load candidate map and all S1 IDs
    all_test_s1 = load_test_s1_ids()
    cand_map = load_test_candidates()

    # Build results: s1_id → list of matched candidate IDs
    matched_df = df[df["match"] == 1][["source1_entity_id", "candidate_entity_id"]].copy()

    results = {s1_id: [] for s1_id in all_test_s1}

    for row in tqdm(matched_df.itertuples(index=False), total=len(matched_df), desc="  Building results"):
        s1_id = row.source1_entity_id
        cand_id = row.candidate_entity_id

        # Safety: only include if in candidate set
        if cand_map and s1_id in cand_map and cand_id not in cand_map[s1_id]:
            continue  # skip — not in candidate set (pipeline safety)

        if s1_id in results:
            results[s1_id].append(cand_id)

    # Write output
    print(f"\n[Write] Writing {OUTPUT_FILE} ...")
    total_written = 0
    non_empty = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for s1_id in sorted(all_test_s1):
            cands = results.get(s1_id, [])
            cands_dedup = sorted(set(cands))
            cand_str = ",".join(cands_dedup)
            writer.writerow([s1_id, cand_str])
            total_written += 1
            if cand_str:
                non_empty += 1

    empty = total_written - non_empty
    print(f"  Total S1 entities: {total_written:,}")
    print(f"  With matches:      {non_empty:,}")
    print(f"  No match (empty):  {empty:,}")
    print(f"[Done] → {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
