"""
data_loader.py - Load and validate raw TSV files.

All files are read with sep='\t' as required by the challenge rules.
Never use comma-separated parsing on these files.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

from . import config as cfg

logger = logging.getLogger(__name__)


# ─── Column Schema ────────────────────────────────────────────────────────────

REQUIRED_SOURCE_COLS = [
    cfg.ENTITY_ID_COL,
    cfg.BUSINESS_NAME_COL,
    cfg.BUSINESS_ADDR_COL,
    cfg.COUNTRY_COL,
]

REQUIRED_GT_COLS = [
    cfg.GT_S1_COL,
    cfg.GT_MATCHED_COL,
]


def _load_tsv(path: Path, required_cols: list, name: str) -> pd.DataFrame:
    """Load a single TSV file with validation of required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    logger.info(f"Loading {name} from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Validate columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: Missing required columns {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Fill NaN to empty string
    df = df.fillna("")

    logger.info(f"  → {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


def load_source(path: Path, name: str = "source") -> pd.DataFrame:
    """Load a source TSV (source1/2/3)."""
    return _load_tsv(path, REQUIRED_SOURCE_COLS, name)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load the ground truth TSV.

    Returns a DataFrame with columns:
        source1_entity_id, matched_entity_ids
    The matched_entity_ids column contains the raw comma-separated string
    (may be empty for singleton entities).
    """
    return _load_tsv(path, REQUIRED_GT_COLS, "ground_truth")


# ─── Train data ───────────────────────────────────────────────────────────────

def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training data.

    Returns:
        (source1, source2, source3, ground_truth)
    """
    s1 = load_source(cfg.TRAIN_SOURCE1, "train_source1")
    s2 = load_source(cfg.TRAIN_SOURCE2, "train_source2")
    s3 = load_source(cfg.TRAIN_SOURCE3, "train_source3")
    gt = load_ground_truth(cfg.TRAIN_GT)
    return s1, s2, s3, gt


# ─── Test data ────────────────────────────────────────────────────────────────

def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test data.

    Returns:
        (source1, source2, source3)
    """
    s1 = load_source(cfg.TEST_SOURCE1, "test_source1")
    s2 = load_source(cfg.TEST_SOURCE2, "test_source2")
    s3 = load_source(cfg.TEST_SOURCE3, "test_source3")
    return s1, s2, s3


# ─── Ground truth parsing ─────────────────────────────────────────────────────

def parse_ground_truth(gt: pd.DataFrame) -> Dict[str, set]:
    """Parse ground truth DataFrame into a dict mapping S1 id → set of matched ids.

    Uses fast vectorized approach — avoids slow iterrows on 2M+ rows.
    """
    result = {}
    s1_col  = cfg.GT_S1_COL
    mat_col = cfg.GT_MATCHED_COL

    s1_ids     = gt[s1_col].str.strip().tolist()
    mat_strs   = gt[mat_col].fillna("").str.strip().tolist()

    for s1_id, matched_str in zip(s1_ids, mat_strs):
        if matched_str:
            result[s1_id] = {x.strip() for x in matched_str.split(",") if x.strip()}
        else:
            result[s1_id] = set()
    return result


# ─── Dataset statistics ───────────────────────────────────────────────────────

def print_dataset_stats(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Optional[pd.DataFrame] = None,
    label: str = "Dataset",
) -> None:
    """Print summary statistics about a dataset."""
    print(f"\n{'='*60}")
    print(f"  {label} Statistics")
    print(f"{'='*60}")
    print(f"  Source 1: {len(s1):,} records")
    print(f"  Source 2: {len(s2):,} records")
    print(f"  Source 3: {len(s3):,} records")
    print(f"  Total:    {len(s1)+len(s2)+len(s3):,} records")

    # Country distribution
    for src_name, src_df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        country_counts = src_df[cfg.COUNTRY_COL].value_counts()
        print(f"\n  {src_name} country distribution:")
        for country, cnt in country_counts.items():
            pct = 100.0 * cnt / len(src_df)
            print(f"    {country:20s}: {cnt:>8,}  ({pct:.1f}%)")

    # Missing value check
    print(f"\n  Missing values:")
    for src_name, src_df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        empty_name = (src_df[cfg.BUSINESS_NAME_COL] == "").sum()
        empty_addr = (src_df[cfg.BUSINESS_ADDR_COL] == "").sum()
        empty_country = (src_df[cfg.COUNTRY_COL] == "").sum()
        print(
            f"    {src_name}: name={empty_name:,} addr={empty_addr:,} country={empty_country:,} empty"
        )

    if gt is not None:
        gt_dict = parse_ground_truth(gt)
        n_singletons = sum(1 for v in gt_dict.values() if not v)
        n_with_matches = sum(1 for v in gt_dict.values() if v)
        total_matches = sum(len(v) for v in gt_dict.values())
        print(f"\n  Ground Truth:")
        print(f"    Total S1 entities:      {len(gt_dict):,}")
        print(f"    Entities w/ matches:    {n_with_matches:,}")
        print(f"    Singleton entities:     {n_singletons:,}")
        print(f"    Total matched pairs:    {total_matches:,}")
        if n_with_matches > 0:
            avg = total_matches / n_with_matches
            print(f"    Avg matches per entity: {avg:.2f}")

    print(f"{'='*60}\n")
