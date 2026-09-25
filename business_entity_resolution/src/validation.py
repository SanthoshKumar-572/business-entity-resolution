"""
validation.py - Input data validation for TSV files.

Checks data quality and prints warnings about potential issues.
Does NOT modify the original data.
"""

import logging
from typing import Optional

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


def validate_source_file(df: pd.DataFrame, name: str, expected_prefix: str) -> bool:
    """Validate a source TSV DataFrame.

    Returns True if valid, raises or warns for issues.
    """
    ok = True

    # Check required columns
    required = [cfg.ENTITY_ID_COL, cfg.BUSINESS_NAME_COL, cfg.BUSINESS_ADDR_COL, cfg.COUNTRY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"{name}: Missing required columns: {missing}")
        return False

    # Check entity_id prefix
    wrong_prefix = df[cfg.ENTITY_ID_COL].str.startswith(expected_prefix)
    if not wrong_prefix.all():
        n_wrong = (~wrong_prefix).sum()
        logger.warning(f"{name}: {n_wrong} entity_ids do not start with '{expected_prefix}'")
        ok = False

    # Check for duplicate entity_ids
    n_dupes = df[cfg.ENTITY_ID_COL].duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"{name}: {n_dupes} duplicate entity_ids found")
        ok = False

    # Check for empty names
    empty_names = (df[cfg.BUSINESS_NAME_COL].fillna("").str.strip() == "").sum()
    if empty_names > 0:
        logger.warning(f"{name}: {empty_names} records with empty business_name")

    # Check for empty addresses
    empty_addrs = (df[cfg.BUSINESS_ADDR_COL].fillna("").str.strip() == "").sum()
    if empty_addrs > 0:
        logger.warning(f"{name}: {empty_addrs} records with empty business_address")

    # Check for empty countries
    empty_countries = (df[cfg.COUNTRY_COL].fillna("").str.strip() == "").sum()
    if empty_countries > 0:
        logger.warning(f"{name}: {empty_countries} records with empty country")

    if ok:
        logger.info(f"{name}: Validation PASSED ({len(df):,} records)")
    else:
        logger.warning(f"{name}: Validation completed WITH WARNINGS")

    return ok


def validate_ground_truth(gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> bool:
    """Validate the ground truth file."""
    ok = True

    s1_ids = set(s1[cfg.ENTITY_ID_COL])
    s2_ids = set(s2[cfg.ENTITY_ID_COL])
    s3_ids = set(s3[cfg.ENTITY_ID_COL])
    valid_match_ids = s2_ids | s3_ids

    # All GT S1 IDs should be in source1
    gt_s1_ids = set(gt[cfg.GT_S1_COL])
    unknown_s1 = gt_s1_ids - s1_ids
    if unknown_s1:
        logger.warning(f"GT: {len(unknown_s1)} S1 IDs not in source1: {list(unknown_s1)[:3]}")
        ok = False

    # All matched IDs should be in S2/S3
    invalid_matches = 0
    for _, row in gt.iterrows():
        matched_str = str(row[cfg.GT_MATCHED_COL]).strip()
        if not matched_str:
            continue
        for mid in matched_str.split(","):
            mid = mid.strip()
            if mid and mid not in valid_match_ids:
                invalid_matches += 1

    if invalid_matches > 0:
        logger.warning(f"GT: {invalid_matches} matched IDs not in S2/S3")
        ok = False

    if ok:
        logger.info(f"Ground truth: Validation PASSED ({len(gt):,} rows)")
    return ok
