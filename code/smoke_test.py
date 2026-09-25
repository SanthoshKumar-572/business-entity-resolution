"""
smoke_test.py — Quick end-to-end pipeline test on 10K rows.

Creates tiny sample files and runs through all pipeline stages
to catch any bugs before running the full dataset.

Usage:
  python3 code/smoke_test.py
"""

import os
import sys
import subprocess
import pandas as pd
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

SAMPLE_DIR = os.path.join(ROOT, "output", "smoke_test")
os.makedirs(SAMPLE_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "output")

PYTHON = sys.executable
SAMPLE_S1 = 5_000
SAMPLE_S2 = 10_000
SAMPLE_S3 = 10_000


def sample_file(source_path: str, dest_path: str, n: int):
    """Sample n rows from a TSV and save to dest."""
    if os.path.isfile(dest_path):
        print(f"  Already exists: {os.path.basename(dest_path)}")
        return
    print(f"  Sampling {n:,} rows from {os.path.basename(source_path)} ...")
    df = pd.read_csv(source_path, sep="\t", dtype=str, nrows=n + 1)
    df = df.iloc[:n]
    df.to_csv(dest_path, sep="\t", index=False)
    print(f"  Saved {len(df):,} rows → {dest_path}")


def run(cmd: str, cwd: str = ROOT) -> bool:
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Smoke Test — 10K rows end-to-end")
    print("=" * 60)

    from normalize import normalize_name, normalize_address, normalize_country, get_blocking_keys
    from collections import defaultdict
    import csv

    # Create sample files
    print("\n[1] Creating sample files ...")
    s1_sample = os.path.join(SAMPLE_DIR, "s1_sample.tsv")
    s2_sample = os.path.join(SAMPLE_DIR, "s2_sample.tsv")
    s3_sample = os.path.join(SAMPLE_DIR, "s3_sample.tsv")
    gt_sample = os.path.join(SAMPLE_DIR, "gt_sample.tsv")

    sample_file(os.path.join(TRAIN_DIR, "train_source1.tsv"), s1_sample, SAMPLE_S1)
    sample_file(os.path.join(TRAIN_DIR, "train_source2.tsv"), s2_sample, SAMPLE_S2)
    sample_file(os.path.join(TRAIN_DIR, "train_source3.tsv"), s3_sample, SAMPLE_S3)

    # Sample ground truth (only for the sampled S1 IDs)
    s1_ids = set(pd.read_csv(s1_sample, sep="\t", dtype=str)["entity_id"].tolist())
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str, nrows=50_000)
    gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids)]
    gt_df.to_csv(gt_sample, sep="\t", index=False)
    print(f"  GT sample: {len(gt_df):,} rows")

    # Build a small index
    print("\n[2] Building blocking index ...")
    index = defaultdict(set)
    for sfile, label in [(s2_sample, "S2"), (s3_sample, "S3")]:
        df = pd.read_csv(sfile, sep="\t", dtype=str).fillna("")
        for _, row in df.iterrows():
            eid = row["entity_id"]
            nn = normalize_name(row["business_name"])
            na = normalize_address(row["business_address"])
            nc = normalize_country(row["country"])
            for kt, kv in get_blocking_keys(nn, na, nc):
                index[f"{kt}::{kv}"].add(eid)
        print(f"  {label}: {len(df):,} entities indexed")

    # Generate candidates
    print("\n[3] Generating candidates ...")
    cand_file = os.path.join(SAMPLE_DIR, "candidates_sample.tsv")
    pairs = []
    s1_df = pd.read_csv(s1_sample, sep="\t", dtype=str).fillna("")
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        nn = normalize_name(row["business_name"])
        na = normalize_address(row["business_address"])
        nc = normalize_country(row["country"])
        cands = set()
        for kt, kv in get_blocking_keys(nn, na, nc):
            cands.update(index.get(f"{kt}::{kv}", set()))
        for c in cands:
            pairs.append((s1_id, c))

    print(f"  Generated {len(pairs):,} candidate pairs for {len(s1_df):,} S1 entities")
    print(f"  Avg candidates per S1: {len(pairs)/len(s1_df):.1f}")

    # Check recall
    if len(gt_df) > 0:
        cand_set = set(pairs)
        total_gt = 0
        recall_hits = 0
        for _, row in gt_df.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            if not isinstance(matched, str) or not matched.strip():
                continue
            for m in matched.split(","):
                total_gt += 1
                if (s1_id, m.strip()) in cand_set:
                    recall_hits += 1
        recall = recall_hits / total_gt if total_gt > 0 else 0
        print(f"\n  Blocking recall: {recall:.3f} ({recall_hits}/{total_gt} GT pairs recovered)")

    print("\n[4] Testing feature computation ...")
    from normalize import (
        normalize_name, normalize_address, normalize_country,
        get_name_tokens, get_address_tokens, extract_numbers,
    )
    from rapidfuzz import fuzz

    if pairs:
        s1_id, cand_id = pairs[0]
        s1_row = s1_df[s1_df["entity_id"] == s1_id].iloc[0]
        s1_name = normalize_name(s1_row["business_name"])
        s1_addr = normalize_address(s1_row["business_address"])
        ratio = fuzz.ratio(s1_name, "prime money llc")
        print(f"  Sample pair: {s1_id} ↔ {cand_id}")
        print(f"  Fuzzy name ratio (vs 'prime money llc'): {ratio:.2f}")

    print("\n✓ Smoke test PASSED")
    return 0


if __name__ == "__main__":
    # Guard against recursive import issue
    import importlib.util
    spec = importlib.util.spec_from_file_location("normalize",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalize.py"))
    sys.exit(main())
