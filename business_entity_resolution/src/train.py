"""
train.py - Model training and persistence.

Primary model: LightGBM classifier on engineered similarity features.
Falls back to XGBoost or LogisticRegression if LightGBM is unavailable.

Key design choices:
- Entity-level train/val split to prevent leakage
- Class imbalance handled via scale_pos_weight
- Early stopping on validation logloss
- Feature importance logging
- F0.5-optimized threshold selection
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from . import config as cfg

logger = logging.getLogger(__name__)


# ─── Model selection ─────────────────────────────────────────────────────────

def _get_model():
    """Return the best available gradient-boosting classifier."""
    try:
        import lightgbm as lgb
        logger.info("Using LightGBM")
        return "lightgbm"
    except ImportError:
        pass

    try:
        import xgboost as xgb
        logger.info("Using XGBoost (LightGBM unavailable)")
        return "xgboost"
    except ImportError:
        pass

    logger.warning("Neither LightGBM nor XGBoost available; using LogisticRegression")
    return "logistic"


# ─── LightGBM trainer ────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
):
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = max(1.0, float(n_neg) / max(n_pos, 1))
    logger.info(f"LightGBM scale_pos_weight: {scale_pos_weight:.2f}")

    params = cfg.TRAINING["lgb_params"].copy()
    n_estimators = params.pop("n_estimators")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        **params,
    )

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        callbacks = [
            lgb.early_stopping(
                cfg.TRAINING["early_stopping_rounds"], verbose=False
            ),
            lgb.log_evaluation(100),
        ]
        fit_kwargs["callbacks"] = callbacks

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(f"LightGBM trained: {model.best_iteration_} iterations")

    # Feature importance
    if feature_names and hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: -x[1],
        )
        logger.info("Top 20 features:")
        for feat, imp in importances[:20]:
            logger.info(f"  {feat:40s}: {imp:.4f}")

    return model


# ─── XGBoost trainer ─────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
):
    """Train an XGBoost classifier."""
    import xgboost as xgb

    params = cfg.TRAINING["xgb_params"].copy()
    n_estimators = params.pop("n_estimators")
    early_stopping = cfg.TRAINING["early_stopping_rounds"]

    # Compute scale_pos_weight from data
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = max(1, n_neg // n_pos)
    logger.info(f"XGBoost scale_pos_weight: {params.get('scale_pos_weight')}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping if X_val is not None else None,
        **params,
    )

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = 100

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(f"XGBoost trained: {getattr(model, 'best_ntree_limit', n_estimators)} trees")

    # Feature importance
    if feature_names and hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: -x[1],
        )
        logger.info("Top 20 features:")
        for feat, imp in importances[:20]:
            logger.info(f"  {feat:40s}: {imp:.4f}")

    return model


# ─── Logistic Regression fallback ────────────────────────────────────────────

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
):
    """Train a Logistic Regression classifier (fallback)."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)
    logger.info("Logistic Regression training complete")

    # Wrap scaler + model together
    return {"model": model, "scaler": scaler, "type": "logistic"}


# ─── Unified training interface ───────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
):
    """Train the best available model and return it."""
    model_type = _get_model()

    logger.info(
        f"Training {model_type} on {len(y_train):,} samples "
        f"({int(y_train.sum()):,} positive)"
    )

    if model_type == "lightgbm":
        return train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
    elif model_type == "xgboost":
        return train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    else:
        return train_logistic(X_train, y_train, X_val, y_val, feature_names)


# ─── Predict probabilities ────────────────────────────────────────────────────

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return P(match=1) for feature matrix X."""
    if isinstance(model, dict) and model.get("type") == "logistic":
        scaler = model["scaler"]
        clf    = model["model"]
        X_sc   = scaler.transform(X)
        return clf.predict_proba(X_sc)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


# ─── Save / Load model ────────────────────────────────────────────────────────

def save_model(model, feature_names: List[str], path=None) -> None:
    """Save model and feature names to disk."""
    model_path = path or cfg.MODEL_PATH
    feat_path  = cfg.FEATURE_NAMES_PATH

    # LightGBM
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(model_path))
        logger.info(f"LightGBM model saved to {model_path}")
    elif hasattr(model, "save_model"):
        # XGBoost
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
    else:
        joblib.dump(model, str(model_path) + ".joblib")
        logger.info(f"Model saved to {str(model_path)}.joblib")

    # Save feature names
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))
    logger.info(f"Feature names saved to {feat_path}")


def load_model(path=None):
    """Load model from disk."""
    import lightgbm as lgb
    model_path = str(path or cfg.MODEL_PATH)

    # Try LightGBM
    try:
        if os.path.exists(model_path):
            booster = lgb.Booster(model_file=model_path)
            # Wrap in a classifier-like object
            class _LGBWrapper:
                def __init__(self, b):
                    self._b = b
                def predict_proba(self, X):
                    p = self._b.predict(X)
                    return np.column_stack([1 - p, p])
            logger.info(f"LightGBM model loaded from {model_path}")
            return _LGBWrapper(booster)
    except Exception:
        pass

    # Try XGBoost
    try:
        import xgboost as xgb
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"XGBoost model loaded from {model_path}")
            return model
    except (ImportError, Exception):
        pass

    # Try joblib
    joblib_path = model_path + ".joblib"
    if os.path.exists(joblib_path):
        model = joblib.load(joblib_path)
        logger.info(f"Joblib model loaded from {joblib_path}")
        return model

    raise FileNotFoundError(f"No model found at {model_path} or {joblib_path}")


def load_feature_names(path=None) -> List[str]:
    """Load feature names from disk."""
    feat_path = path or cfg.FEATURE_NAMES_PATH
    with open(feat_path) as f:
        return [line.strip() for line in f if line.strip()]
