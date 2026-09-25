"""
config.py - Central configuration for the Business Entity Resolution pipeline.

All tunable parameters, paths, and constants are defined here.
Modify this file to adjust the pipeline behavior without touching core logic.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[1]          # business_entity_resolution/
STUDENT_RESOURCE_DIR = ROOT_DIR.parent                  # student_resource/

DATASET_DIR    = STUDENT_RESOURCE_DIR / "dataset"
TRAIN_DIR      = DATASET_DIR / "train"
TEST_DIR       = DATASET_DIR / "test"
OUTPUT_DIR     = ROOT_DIR / "output"
MODELS_DIR     = ROOT_DIR / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CACHE_DIR      = ROOT_DIR / "cache"
LOGS_DIR       = ROOT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─── File Paths ───────────────────────────────────────────────────────────────

TRAIN_SOURCE1   = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2   = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3   = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT        = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1    = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2    = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3    = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING   = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# Output directory for THIS module (used by run_pipeline.py)
STUDENT_OUTPUT_DIR = STUDENT_RESOURCE_DIR / "output"
STUDENT_OUTPUT_MATCHING   = STUDENT_OUTPUT_DIR / "matching_results.tsv"
STUDENT_OUTPUT_CANDIDATES = STUDENT_OUTPUT_DIR / "candidate_pairs.tsv"

MODEL_PATH          = MODELS_DIR / "lgb_model.txt"
THRESHOLD_PATH      = MODELS_DIR / "threshold.txt"
FEATURE_NAMES_PATH  = MODELS_DIR / "feature_names.txt"
EXPERIMENTS_CSV     = EXPERIMENTS_DIR / "results.csv"

# Cache files for restartable pipeline
TRAIN_CANDIDATES_CACHE  = CACHE_DIR / "train_candidates.pkl"
VAL_CANDIDATES_CACHE    = CACHE_DIR / "val_candidates.pkl"
TEST_CANDIDATES_CACHE   = CACHE_DIR / "test_candidates.pkl"
TRAIN_FEATURES_CACHE    = CACHE_DIR / "train_features.pkl"
VAL_FEATURES_CACHE      = CACHE_DIR / "val_features.pkl"
TFIDF_CACHE             = CACHE_DIR / "tfidf_computers.pkl"

VALIDATE_SCRIPT = STUDENT_RESOURCE_DIR / "utils" / "validate_submission.py"

# ─── Data Schema ──────────────────────────────────────────────────────────────

ENTITY_ID_COL       = "entity_id"
BUSINESS_NAME_COL   = "business_name"
BUSINESS_ADDR_COL   = "business_address"
COUNTRY_COL         = "country"

GT_S1_COL           = "source1_entity_id"
GT_MATCHED_COL      = "matched_entity_ids"

SOURCE_PREFIXES     = {"S1", "S2", "S3"}
MATCH_PREFIXES      = {"S2", "S3"}

# ─── Blocking Configuration ───────────────────────────────────────────────────

BLOCKING = {
    # Maximum candidates per S1 entity (safety cap to limit memory)
    "max_candidates_per_s1": 500,

    # TF-IDF blocking: top-k candidates per S1 from cosine similarity
    "tfidf_top_k": 80,

    # Character n-gram sizes for blocking
    "char_ngram_sizes": [3, 4],

    # Minimum token length to be used in blocking
    "min_token_len": 2,

    # How many trigrams from S1 name must match S2/S3 name for blocking
    "ngram_overlap_threshold": 1,

    # Whether to use country-based blocking
    "use_country_blocking": True,
}

# ─── Feature Engineering ──────────────────────────────────────────────────────

FEATURES = {
    # Character n-gram sizes for similarity features
    "char_ngram_sizes": [2, 3, 4],

    # TF-IDF vectorizer settings
    "tfidf_max_features": 50000,
    "tfidf_min_df": 1,

    # Jaro-Winkler prefix weight
    "jaro_winkler_prefix_weight": 0.1,
}

# ─── Training ─────────────────────────────────────────────────────────────────

TRAINING = {
    # Fraction of S1 entities to hold out for validation
    "validation_fraction": 0.2,

    # Random seed for reproducibility
    "random_seed": 42,

    # Negative sampling ratio (negatives per positive) for training balance
    # None = use all negatives
    "negative_sample_ratio": 8,

    # LightGBM hyperparameters (better than XGBoost for entity resolution)
    "lgb_params": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    },

    # XGBoost hyperparameters (fallback)
    "xgb_params": {
        "n_estimators": 800,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 3,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    },

    # Early stopping rounds
    "early_stopping_rounds": 50,

    # Feature computation chunk size
    "feature_chunk_size": 100_000,
}

# ─── Threshold Optimization ───────────────────────────────────────────────────

THRESHOLD = {
    # Thresholds to search over
    "search_thresholds": [
        0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85,
        0.90, 0.92, 0.94, 0.95,
        0.97, 0.98,
    ],

    # Default threshold if no validation data available
    "default_threshold": 0.70,

    # F-beta beta parameter (0.5 = precision-heavy)
    "fbeta": 0.5,
}

# ─── Normalization ────────────────────────────────────────────────────────────

NORMALIZATION = {
    # Legal suffixes to normalize (maps → canonical form)
    "legal_suffixes": {
        r"\bprivate\s+limited\b": "pvt ltd",
        r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
        r"\bp\.?\s*ltd\.?\b": "pvt ltd",
        r"\blimited\b": "ltd",
        r"\bincorporated\b": "inc",
        r"\bcorporation\b": "corp",
        r"\bcompany\b": "co",
        r"\bllc\.?\b": "llc",
        r"\bllp\.?\b": "llp",
        r"\bplc\.?\b": "plc",
        r"\bsociété\b": "societe",
        r"\bsarl\b": "sarl",
        r"\bsas\b": "sas",
        r"\bsa\.?\b": "sa",
    },

    # Common abbreviations to expand for names
    "name_abbrevs": {
        r"\&": "and",
        r"\bbros\.?\b": "brothers",
        r"\bsons\.?\b": "sons",
        r"\bintl\.?\b": "international",
        r"\bint\.?\b": "international",
        r"\bmfg\.?\b": "manufacturing",
        r"\bsvc\.?\b": "services",
        r"\bsvcs\.?\b": "services",
        r"\btech\.?\b": "technologies",
        r"\bgrp\.?\b": "group",
        r"\bgp\.?\b": "group",
        r"\benterp\.?\b": "enterprises",
        r"\benterps\.?\b": "enterprises",
        r"\btrading\.?\b": "trading",
        r"\binds\.?\b": "industries",
        r"\bind\.?\b": "industries",
        r"\bmgmt\.?\b": "management",
        r"\bhospital\b": "hospital",
        r"\bhosp\.?\b": "hospital",
    },

    # Address abbreviation expansions
    "addr_abbrevs": {
        r"\bst\.?\b": "street",
        r"\brd\.?\b": "road",
        r"\bave\.?\b": "avenue",
        r"\bavenue\b": "avenue",
        r"\bblvd\.?\b": "boulevard",
        r"\bdr\.?\b": "drive",
        r"\bln\.?\b": "lane",
        r"\bpl\.?\b": "place",
        r"\bct\.?\b": "court",
        r"\bflr\.?\b": "floor",
        r"\bfl\.?\b": "floor",
        r"\bno\.?\s*": "no ",
        r"\bsector\b": "sector",
        r"\bsec\.?\b": "sector",
        r"\bph\.?\b": "phase",
        r"\bnear\b": "near",
        r"\bopp\.?\b": "opposite",
        r"\bopposite\b": "opposite",
        r"\bnagar\b": "nagar",
        r"\bcolony\b": "colony",
    },
}

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
