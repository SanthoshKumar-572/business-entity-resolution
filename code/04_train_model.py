"""
04_train_model.py
==================
Train a LightGBM classifier on features_train.parquet.

Pipeline:
  1. Load features_train.parquet
  2. Split by Source 1 entity (avoid leakage)
  3. Train LightGBM with class_weight handling
  4. Evaluate validation F0.5 at multiple thresholds
  5. Select best threshold
  6. Save model to model/lgbm_model.txt
  7. Save threshold to model/threshold.txt
  8. Save feature importance plot metadata to model/feature_importance.csv
"""

import os
import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(OUTPUT_DIR, "features_train.parquet")
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_model.txt")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "threshold.txt")
FEAT_IMPORTANCE_FILE = os.path.join(MODEL_DIR, "feature_importance.csv")
METRICS_FILE = os.path.join(MODEL_DIR, "validation_metrics.json")

# Feature columns (exclude metadata + label)
META_COLS = {"source1_entity_id", "candidate_entity_id", "label"}

# Thresholds to test
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]

VAL_FRAC = 0.2   # Fraction of S1 entities for validation
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def pregroup_val_data(df_val: pd.DataFrame) -> dict:
    """Pre-group validation data by S1 entity for fast evaluation."""
    groups = {}
    for s1_id, cand_id, prob, label in zip(
        df_val["source1_entity_id"],
        df_val["candidate_entity_id"],
        df_val["prob"],
        df_val["label"],
    ):
        if s1_id not in groups:
            groups[s1_id] = []
        groups[s1_id].append((cand_id, prob, label))
    return groups


def compute_f05_fast(pregrouped: dict, threshold: float) -> dict:
    """
    Compute F0.5 at entity level (macro-averaged over S1 entities).

    For each S1 entity:
      - predicted = set of candidates where prob >= threshold
      - actual    = set of candidates where label == 1
      - entity f0.5 = (1.25 * p * r) / (0.25 * p + r) if denom > 0, else 1 if both empty
    """
    beta2 = 0.25
    entity_scores = []
    precisions = []
    recalls = []

    for s1_id, items in pregrouped.items():
        predicted = [cand_id for cand_id, prob, _ in items if prob >= threshold]
        actual = [cand_id for cand_id, _, label in items if label == 1]

        if not predicted and not actual:
            entity_f05 = 1.0
            precisions.append(1.0)
            recalls.append(1.0)
        elif not predicted:
            entity_f05 = 0.0
            precisions.append(0.0)
            recalls.append(0.0)
        elif not actual:
            entity_f05 = 0.0
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            set_pred = set(predicted)
            set_act = set(actual)
            tp = len(set_pred & set_act)
            p = tp / len(set_pred)
            r = tp / len(set_act)
            denom = beta2 * p + r
            entity_f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
            precisions.append(p)
            recalls.append(r)

        entity_scores.append(entity_f05)

    return {
        "f05": float(np.mean(entity_scores)) if entity_scores else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_entities": len(pregrouped),
    }


def tune_threshold(df_val: pd.DataFrame, thresholds: list) -> tuple:
    """Evaluate all thresholds on validation set. Returns (best_threshold, metrics_dict)."""
    print("\n[Threshold Tuning]")
    print(f"  {'Threshold':>10} | {'F0.5':>8} | {'Precision':>10} | {'Recall':>8} | {'N Entities':>10}")
    print("  " + "-" * 56)

    pregrouped = pregroup_val_data(df_val)

    best_f05 = -1
    best_threshold = 0.5
    all_metrics = {}

    for thr in thresholds:
        result = compute_f05_fast(pregrouped, thr)
        f05 = result["f05"]
        all_metrics[thr] = result

        marker = " ← best" if f05 > best_f05 else ""
        print(f"  {thr:>10.2f} | {f05:>8.4f} | {result['precision']:>10.4f} | {result['recall']:>8.4f} | {result['n_entities']:>10,}{marker}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thr

    return best_threshold, all_metrics



def main():
    print("=" * 60)
    print("LightGBM Model Training + Threshold Tuning")
    print("=" * 60)

    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    # Load features
    print(f"\n[Data] Loading {FEATURES_FILE} ...")
    if not os.path.isfile(FEATURES_FILE):
        print(f"  ERROR: {FEATURES_FILE} not found. Run 03_feature_engineering.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"  Loaded {len(df):,} rows")

    # Remove unlabeled rows (label=-1 means test)
    df = df[df["label"] >= 0].copy()
    print(f"  After removing test rows: {len(df):,} rows")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    groups = df["source1_entity_id"].values

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    print(f"  Positive pairs: {pos_count:,}, Negative pairs: {neg_count:,}")
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # Group-based train/val split
    print(f"\n[Split] Splitting by S1 entity (val_frac={VAL_FRAC}) ...")
    unique_s1 = np.unique(groups)
    rng = np.random.default_rng(RANDOM_SEED)
    val_s1 = set(rng.choice(unique_s1, size=int(len(unique_s1) * VAL_FRAC), replace=False))
    train_mask = np.array([g not in val_s1 for g in groups])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    df_val = df[val_mask].copy()

    print(f"  Train: {len(X_train):,} pairs, Val: {len(X_val):,} pairs")
    print(f"  Train S1 entities: {(~val_mask).sum() and len(set(groups[train_mask])):,}, Val S1 entities: {len(val_s1):,}")

    # LightGBM training
    print("\n[Train] Training LightGBM ...")
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_cols,
        free_raw_data=True,
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        reference=dtrain,
        feature_name=feature_cols,
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"  Best iteration: {model.best_iteration}")

    # Predict validation probabilities
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    df_val = df_val.copy()
    df_val["prob"] = val_probs

    # Threshold tuning
    best_threshold, all_metrics = tune_threshold(df_val, THRESHOLDS)
    print(f"\n  Best threshold: {best_threshold} (F0.5={all_metrics[best_threshold]['f05']:.4f})")

    # Save model
    model.save_model(MODEL_FILE)
    print(f"\n[Save] Model saved → {MODEL_FILE}")

    # Save threshold
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(best_threshold))
    print(f"[Save] Threshold saved → {THRESHOLD_FILE}")

    # Save feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance("gain"),
        "importance_split": model.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(FEAT_IMPORTANCE_FILE, index=False)
    print(f"[Save] Feature importance saved → {FEAT_IMPORTANCE_FILE}")
    print("\n  Top 10 features:")
    print(importance.head(10).to_string(index=False))

    # Save metrics
    metrics_out = {
        "best_threshold": best_threshold,
        "best_f05": all_metrics[best_threshold]["f05"],
        "all_thresholds": {str(k): v for k, v in all_metrics.items()},
        "best_iteration": model.best_iteration,
        "train_pairs": int(len(X_train)),
        "val_pairs": int(len(X_val)),
        "train_s1_entities": int(len(set(groups[train_mask]))),
        "val_s1_entities": int(len(val_s1)),
        "scale_pos_weight": float(scale_pos_weight),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[Save] Metrics saved → {METRICS_FILE}")

    print("\n[Done] Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
