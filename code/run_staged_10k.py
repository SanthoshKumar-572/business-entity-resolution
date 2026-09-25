"""
run_staged_10k.py
=================
First Staged Benchmark (10,000 S1 Entities):
1. Load 10,000 S1 records from S1_Laptop2.tsv
2. Generate candidates against full Source 2 & Source 3 (using two-stage filtered blocking)
3. Compute candidate reduction, recall, and pair counts
4. Generate features for all candidate pairs
5. Group-based train/validation split by S1 entity (80% train / 20% val)
6. Train initial LightGBM classifier
7. Evaluate F0.5, Precision, Recall across all 13 thresholds:
   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
8. Evaluate performance on zero-match (singletons), 1-match, and multi-match entities
9. Report all timings, RAM, reduction, and metric results for user review
"""

import os
import sys
import gc
import time
import json
import csv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from normalize import (
    normalize_name, normalize_address, normalize_country,
    get_blocking_keys, extract_numbers, LEGAL_STOPWORDS, ADDR_STOPWORDS
)

# Import directly from 03_feature_engineering
import importlib.util
spec = importlib.util.spec_from_file_location("fe", os.path.join(SCRIPT_DIR, "03_feature_engineering.py"))
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)

spec_train = importlib.util.spec_from_file_location("tm", os.path.join(SCRIPT_DIR, "04_train_model.py"))
tm = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(tm)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
S1_FILE = os.path.join(TRAIN_DIR, "S1_Laptop2.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

OUTPUT_DIR = os.path.join(ROOT, "output", "staged_10k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_S1 = 10_000
CHUNK_SIZE = 250_000
MAX_CANDIDATES_PER_BLOCK_KEY = 200
TOP_K_CANDIDATES = 50
MIN_SCORE = 0.10
VAL_FRAC = 0.20
RANDOM_SEED = 42

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]


def token_overlap_score(name_a, name_b, addr_a, addr_b, country_a, country_b):
    if country_a and country_b and country_a != country_b:
        return 0.0
    toks_a = set(t for t in name_a.split() if len(t) >= 3 and t not in LEGAL_STOPWORDS)
    toks_b = set(t for t in name_b.split() if len(t) >= 3 and t not in LEGAL_STOPWORDS)
    if not toks_a: toks_a = set(t for t in name_a.split() if len(t) >= 3)
    if not toks_b: toks_b = set(t for t in name_b.split() if len(t) >= 3)

    name_jac = len(toks_a & toks_b) / len(toks_a | toks_b) if (toks_a or toks_b) else 0.0
    prefix_bonus = 0.15 if name_a[:4] and name_a[:4] == name_b[:4] else 0.0
    nums_a = set(extract_numbers(addr_a))
    nums_b = set(extract_numbers(addr_b))
    addr_num = len(nums_a & nums_b) / len(nums_a | nums_b) if (nums_a and nums_b) else 0.0
    return 0.6 * name_jac + 0.25 * addr_num + prefix_bonus


def build_candidate_index():
    print("\n[Step 1/5] Building Inverted Blocking Index for S2 and S3 ...")
    t0 = time.time()
    index = {}
    entity_records = {}

    for src_file, label in [(S2_FILE, "Source2"), (S3_FILE, "Source3")]:
        reader = pd.read_csv(
            src_file, sep="\t", dtype=str, chunksize=CHUNK_SIZE,
            usecols=["entity_id", "business_name", "business_address", "country"]
        )
        count = 0
        for chunk in tqdm(reader, desc=f"  Indexing {label}"):
            chunk = chunk.fillna("")
            for row in chunk.itertuples(index=False):
                eid = row.entity_id
                nn = normalize_name(row.business_name)
                na = normalize_address(row.business_address)
                nc = normalize_country(row.country)
                entity_records[eid] = (nn, na, nc, row.country)

                for kt, kv in get_blocking_keys(nn, na, nc):
                    k = f"{kt}::{kv}"
                    cands = index.get(k)
                    if cands is None:
                        index[k] = [eid]
                    elif len(cands) < MAX_CANDIDATES_PER_BLOCK_KEY:
                        cands.append(eid)
                count += 1
            del chunk
        print(f"    Loaded {count:,} entities from {label}")

    t_idx = time.time() - t0
    print(f"  Index built in {t_idx:.1f}s: {len(index):,} unique keys, {len(entity_records):,} candidate records")
    return index, entity_records, t_idx


def generate_candidates_for_s1(index, entity_records):
    print(f"\n[Step 2/5] Generating Candidates for 10K S1 Entities from S1_Laptop2.tsv ...")
    t0 = time.time()
    df_s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=N_S1).fillna("")
    print(f"  Loaded {len(df_s1):,} S1 rows")

    s1_lookup = {}
    candidate_pairs = []
    raw_pairs_count = 0

    for row in tqdm(df_s1.itertuples(index=False), total=len(df_s1), desc="  Generating"):
        s1_id = row.entity_id
        s1_nn = normalize_name(row.business_name)
        s1_na = normalize_address(row.business_address)
        s1_nc = normalize_country(row.country)
        s1_lookup[s1_id] = (s1_nn, s1_na, s1_nc, row.country)

        raw = set()
        for kt, kv in get_blocking_keys(s1_nn, s1_na, s1_nc):
            hits = index.get(f"{kt}::{kv}")
            if hits:
                raw.update(hits)
        raw_pairs_count += len(raw)

        scored = []
        for cid in raw:
            crec = entity_records.get(cid)
            if not crec: continue
            score = token_overlap_score(s1_nn, crec[0], s1_na, crec[1], s1_nc, crec[2])
            if score >= MIN_SCORE:
                scored.append((score, cid))

        scored.sort(reverse=True)
        top = [cid for _, cid in scored[:TOP_K_CANDIDATES]]
        for cid in top:
            candidate_pairs.append((s1_id, cid))

    t_cand = time.time() - t0
    reduction = (1 - len(candidate_pairs) / max(raw_pairs_count, 1)) * 100
    print(f"  Candidates generated in {t_cand:.1f}s")
    print(f"    Raw blocking pairs: {raw_pairs_count:,} ({raw_pairs_count / len(df_s1):.1f}/S1)")
    print(f"    Filtered pairs:     {len(candidate_pairs):,} ({len(candidate_pairs) / len(df_s1):.1f}/S1)")
    print(f"    Reduction ratio:    {reduction:.1f}%")

    return df_s1, s1_lookup, candidate_pairs, t_cand, raw_pairs_count


def compute_features(s1_lookup, cand_lookup, candidate_pairs, ground_truth):
    print(f"\n[Step 3/5] Computing Features for {len(candidate_pairs):,} Pairs ...")
    t0 = time.time()
    feature_rows = []

    for s1_id, cid in tqdm(candidate_pairs, desc="  Feature calculation"):
        feat = fe.compute_features_for_pair(
            s1_id=s1_id,
            cand_id=cid,
            s1_lookup=s1_lookup,
            cand_lookup=cand_lookup,
            ground_truth=ground_truth,
        )
        feature_rows.append(feat)

    df_feats = pd.DataFrame(feature_rows)
    t_feat = time.time() - t0
    pos_count = df_feats["label"].sum()
    neg_count = len(df_feats) - pos_count
    print(f"  Features computed in {t_feat:.1f}s ({len(df_feats):,} rows, {len(df_feats.columns)} cols)")
    print(f"    Positives: {pos_count:,} | Negatives: {neg_count:,} | Imbalance ratio: 1:{neg_count/max(pos_count,1):.1f}")
    return df_feats, t_feat


def run_experiment():
    print("=" * 65)
    print("STAGED BENCHMARK 1: 10,000 S1 ENTITIES (PERSON 2 PIPELINE)")
    print("=" * 65)

    index, entity_records, t_idx = build_candidate_index()
    df_s1, s1_lookup, candidate_pairs, t_cand, raw_count = generate_candidates_for_s1(index, entity_records)

    # Free index to save RAM
    del index
    gc.collect()

    # Load ground truth
    print("\nLoading Ground Truth ...")
    gt = fe.load_ground_truth()

    # Compute recall of candidates for our 10K S1
    gt_s1_pairs = 0
    gt_s1_hits = 0
    cands_by_s1 = {}
    for s1_id, cid in candidate_pairs:
        if s1_id not in cands_by_s1:
            cands_by_s1[s1_id] = set()
        cands_by_s1[s1_id].add(cid)

    for s1_id in df_s1["entity_id"]:
        matches = gt.get(s1_id, set())
        for m in matches:
            gt_s1_pairs += 1
            if m in cands_by_s1.get(s1_id, set()):
                gt_s1_hits += 1

    cand_recall = gt_s1_hits / max(gt_s1_pairs, 1)
    print(f"  Candidate Generation Recall on 10K S1: {gt_s1_hits}/{gt_s1_pairs} = {cand_recall*100:.2f}%")

    # Feature computation
    df_feats, t_feat = compute_features(s1_lookup, entity_records, candidate_pairs, gt)

    # Train / Val Split by S1 entity
    print(f"\n[Step 4/5] Training LightGBM Model (80% train / 20% val) ...")
    t0_train = time.time()
    unique_s1 = df_s1["entity_id"].values
    rng = np.random.default_rng(RANDOM_SEED)
    val_s1 = set(rng.choice(unique_s1, size=int(len(unique_s1) * VAL_FRAC), replace=False))

    feature_cols = [c for c in df_feats.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]

    train_mask = ~df_feats["source1_entity_id"].isin(val_s1)
    val_mask = df_feats["source1_entity_id"].isin(val_s1)

    df_train = df_feats[train_mask]
    df_val = df_feats[val_mask].copy()

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.int32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["label"].values.astype(np.int32)

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    print(f"  Train: {len(X_train):,} pairs ({y_train.sum():,} pos) | Val: {len(X_val):,} pairs ({y_val.sum():,} pos)")
    print(f"  Validation S1 entities: {len(val_s1):,}")

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "scale_pos_weight": scale_pos,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    t_train = time.time() - t0_train
    print(f"  Model trained in {t_train:.1f}s (best iteration: {model.best_iteration})")

    # Evaluate probabilities on validation set
    df_val["prob"] = model.predict(X_val, num_iteration=model.best_iteration)

    # Threshold tuning
    print(f"\n[Step 5/5] Threshold Tuning and Entity-Level F0.5 Evaluation ...")
    best_thr, all_metrics = tm.tune_threshold(df_val, THRESHOLDS)

    # Detailed breakdown for best threshold
    pregrouped = tm.pregroup_val_data(df_val)
    singletons = []
    single_match = []
    multi_match = []

    for s1_id, items in pregrouped.items():
        actual = [cid for cid, _, lbl in items if lbl == 1]
        preds = [cid for cid, pr, _ in items if pr >= best_thr]
        tp = len(set(preds) & set(actual))
        p = tp / len(preds) if preds else (1.0 if not actual else 0.0)
        r = tp / len(actual) if actual else (1.0 if not preds else 0.0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else (1.0 if not preds and not actual else 0.0)

        if len(actual) == 0:
            singletons.append(f05)
        elif len(actual) == 1:
            single_match.append(f05)
        else:
            multi_match.append(f05)

    print("\n" + "=" * 65)
    print(f"BENCHMARK SUMMARY (10,000 S1 ENTITIES)")
    print("=" * 65)
    print(f"  Candidate Pairs:           {len(candidate_pairs):,} ({len(candidate_pairs)/N_S1:.1f} per S1)")
    print(f"  Candidate Reduction:       {(1 - len(candidate_pairs)/max(raw_count,1))*100:.1f}%")
    print(f"  Candidate Blocking Recall: {cand_recall*100:.2f}%")
    print(f"  Index Build Time:          {t_idx:.1f}s")
    print(f"  Candidate Gen Time:        {t_cand:.1f}s")
    print(f"  Feature Eng Time:          {t_feat:.1f}s")
    print(f"  Model Training Time:       {t_train:.1f}s")
    print(f"  Best Threshold:            {best_thr:.2f}")
    print(f"  Validation F0.5 (overall): {all_metrics[best_thr]['f05']:.4f}")
    print(f"  Validation Precision:      {all_metrics[best_thr]['precision']:.4f}")
    print(f"  Validation Recall:         {all_metrics[best_thr]['recall']:.4f}")
    print(f"  Performance by Entity Type:")
    print(f"    - Singletons (no match): {np.mean(singletons):.4f} ({len(singletons):,} entities)")
    print(f"    - Single match:          {np.mean(single_match):.4f} ({len(single_match):,} entities)")
    print(f"    - Multiple matches:      {np.mean(multi_match):.4f} ({len(multi_match):,} entities)")
    print("=" * 65)

    # Save summary results
    results = {
        "n_s1": N_S1,
        "n_candidate_pairs": len(candidate_pairs),
        "candidates_per_s1": len(candidate_pairs) / N_S1,
        "reduction_pct": (1 - len(candidate_pairs)/max(raw_count,1))*100,
        "blocking_recall": cand_recall,
        "best_threshold": best_thr,
        "best_f05": all_metrics[best_thr]["f05"],
        "precision": all_metrics[best_thr]["precision"],
        "recall": all_metrics[best_thr]["recall"],
        "singletons_f05": float(np.mean(singletons)),
        "single_match_f05": float(np.mean(single_match)),
        "multi_match_f05": float(np.mean(multi_match)),
        "threshold_metrics": {str(k): v for k, v in all_metrics.items()},
        "timing": {
            "index_s": t_idx,
            "candidate_gen_s": t_cand,
            "feature_eng_s": t_feat,
            "training_s": t_train,
        }
    }
    with open(os.path.join(OUTPUT_DIR, "staged_10k_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment()
