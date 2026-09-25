"""
diagnose_blocking.py - Diagnose blocking recall at different scales.

Run from student_resource/:
    python diagnose_blocking.py

This tests blocking recall by loading the FULL S2+S3 files and checking
which S1→GT matches appear in the candidates.
"""
import sys
import logging
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def diagnose(n_s1=2000):
    """Test blocking recall using full S2/S3 but sample of S1."""
    from business_entity_resolution.src import config as cfg
    from business_entity_resolution.src.data_loader import parse_ground_truth
    from business_entity_resolution.src.normalization import normalize_dataframe
    from business_entity_resolution.src.blocking import generate_candidates
    from business_entity_resolution.src.labeling import check_blocking_recall

    logger.info(f"Blocking Recall Diagnosis: n_s1={n_s1}")
    t0 = time.time()

    # Load GT
    logger.info("Loading ground truth...")
    gt_df = pd.read_csv(cfg.TRAIN_GT, sep="\t", dtype=str).fillna("")
    gt_dict = parse_ground_truth(gt_df)

    # Find S1 entities with matches to test recall properly
    s1_with_matches = [k for k, v in gt_dict.items() if v][:n_s1]
    logger.info(f"Selected {len(s1_with_matches)} S1 entities with matches")

    # Load S1 for these entities
    logger.info("Loading S1...")
    s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", dtype=str).fillna("")
    s1 = s1_full[s1_full["entity_id"].isin(set(s1_with_matches))].reset_index(drop=True)
    del s1_full
    gc.collect()

    # Load FULL S2 and S3
    logger.info("Loading FULL S2...")
    s2 = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", dtype=str).fillna("")
    logger.info(f"  S2: {len(s2):,} records")

    logger.info("Loading FULL S3...")
    s3 = pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", dtype=str).fillna("")
    logger.info(f"  S3: {len(s3):,} records")

    # Normalize
    logger.info("Normalizing...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    gc.collect()

    # Generate candidates
    logger.info("Generating candidates...")
    candidates = generate_candidates(s1, s2, s3)

    # Check recall
    gt_subset = {k: v for k, v in gt_dict.items() if k in set(s1_with_matches)}
    recall = check_blocking_recall(candidates, gt_subset, "full_diagnosis")

    # Stats
    cand_sizes = [len(v) for v in candidates.values()]
    non_empty = sum(1 for s in cand_sizes if s > 0)
    total = sum(cand_sizes)
    elapsed = time.time() - t0

    logger.info("\n" + "="*60)
    logger.info("DIAGNOSIS RESULTS:")
    logger.info(f"  S1 entities tested: {len(s1_with_matches)}")
    logger.info(f"  Blocking recall:    {recall:.4f} ({recall*100:.1f}%)")
    logger.info(f"  Candidate coverage: {non_empty}/{len(candidates)}")
    logger.info(f"  Total candidates:   {total:,}")
    logger.info(f"  Avg per S1:         {total/max(len(candidates),1):.1f}")
    logger.info(f"  Median per S1:      {sorted(cand_sizes)[len(cand_sizes)//2]}")
    logger.info(f"  Max per S1:         {max(cand_sizes) if cand_sizes else 0}")
    logger.info(f"  Runtime:            {elapsed/60:.1f} minutes")
    logger.info("="*60)

    return recall


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-s1", type=int, default=2000,
                        help="Number of S1 entities to test (default: 2000)")
    args = parser.parse_args()
    diagnose(n_s1=args.n_s1)
