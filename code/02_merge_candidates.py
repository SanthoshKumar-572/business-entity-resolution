"""
02_merge_candidates.py
=======================
Merge candidates_laptop1.tsv (from Person 1) and candidates_laptop2.tsv
into a single output/candidate_pairs.tsv.

Expected input columns (both files):
    source1_entity_id   candidate_entity_id

Output candidate_pairs.tsv has the format required by the validator:
    source1_entity_id   candidate_entity_ids   (comma-separated list)

Rules:
  - Remove duplicate pairs.
  - Verify every candidate is S2- or S3- prefixed.
  - One row per Source 1 entity (S1 entities with no candidates still get a row).
  - candidate_entity_ids is empty string for no candidates.
"""

import os
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(ROOT, "output")
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")

LAPTOP1_CANDIDATES = os.path.join(OUTPUT_DIR, "candidates_laptop1.tsv")
LAPTOP2_CANDIDATES = os.path.join(OUTPUT_DIR, "candidates_laptop2.tsv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

CHUNK_SIZE = 500_000


def read_candidate_file_chunked(filepath: str, label: str) -> dict:
    """
    Read a candidates TSV with columns [source1_entity_id, candidate_entity_id].
    Returns dict: s1_id → set(candidate_ids).
    """
    print(f"  Reading {label} from {filepath} ...")
    pairs = defaultdict(set)
    bad_prefix = 0

    if not os.path.isfile(filepath):
        print(f"  WARNING: {filepath} not found — skipping.")
        return pairs

    reader = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    )

    for chunk in tqdm(reader, desc=f"    {label}"):
        chunk = chunk.fillna("")
        for _, row in chunk.iterrows():
            s1_id = str(row.get("source1_entity_id", "")).strip()
            cand_id = str(row.get("candidate_entity_id", "")).strip()
            if not s1_id or not cand_id:
                continue
            if not (cand_id.startswith("S2-") or cand_id.startswith("S3-")):
                bad_prefix += 1
                continue
            pairs[s1_id].add(cand_id)
        del chunk

    if bad_prefix:
        print(f"  WARNING: {bad_prefix} rows with non-S2/S3 candidate IDs removed.")

    total_pairs = sum(len(v) for v in pairs.values())
    print(f"  {label}: {len(pairs):,} S1 entities, {total_pairs:,} candidate pairs")
    return pairs


def load_all_s1_ids(mode: str = "train") -> set:
    """Load all S1 entity IDs from source1 file for a given mode."""
    if mode == "train":
        path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    else:
        path = os.path.join(TEST_DIR, "test_source1.tsv")

    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found.")
        return set()

    print(f"  Loading S1 IDs from {path} ...")
    ids = set()
    reader = pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, usecols=["entity_id"])
    for chunk in reader:
        ids.update(chunk["entity_id"].dropna().tolist())
        del chunk
    print(f"  Loaded {len(ids):,} S1 entities")
    return ids


def main():
    print("=" * 60)
    print("Candidate Merging — laptop1 + laptop2 → candidate_pairs.tsv")
    print("=" * 60)

    # Load both candidate files
    pairs_l1 = read_candidate_file_chunked(LAPTOP1_CANDIDATES, "laptop1")
    pairs_l2 = read_candidate_file_chunked(LAPTOP2_CANDIDATES, "laptop2")

    # Merge
    print("\n[Merge] Combining candidate sets ...")
    merged = defaultdict(set)
    for s1_id, cands in pairs_l1.items():
        merged[s1_id].update(cands)
    for s1_id, cands in pairs_l2.items():
        merged[s1_id].update(cands)

    total = sum(len(v) for v in merged.values())
    print(f"  Merged: {len(merged):,} S1 entities, {total:,} unique candidate pairs")

    # Load all train S1 IDs to ensure every S1 has a row
    all_s1 = load_all_s1_ids("train")

    # For S1 entities with no candidates, add empty entry
    for s1_id in all_s1:
        if s1_id not in merged:
            merged[s1_id] = set()

    # Write output
    print(f"\n[Write] Writing {OUTPUT_FILE} ...")
    rows = []
    for s1_id in tqdm(sorted(merged.keys()), desc="  Building rows"):
        cands = merged[s1_id]
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cands)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    non_empty = df["candidate_entity_ids"].str.len().gt(0).sum()
    print(f"  Wrote {len(df):,} rows ({non_empty:,} with candidates)")
    print(f"[Done] → {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
