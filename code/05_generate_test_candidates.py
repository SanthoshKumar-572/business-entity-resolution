"""
05_generate_test_candidates.py
===============================
Generate candidate pairs for TEST Source 1 entities.

Uses the SAME blocking strategy as training candidate generation.
Produces: output/test_candidate_pairs.tsv

Format (validator-compatible):
    source1_entity_id   candidate_entity_ids   (comma-separated)

Every test S1 entity gets a row (empty if no candidates found).
"""

import os
import sys
import gc
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from normalize import normalize_name, normalize_address, normalize_country, get_blocking_keys

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_candidate_pairs.tsv")

CHUNK_SIZE = 200_000
MAX_CANDIDATES_PER_BLOCK_KEY = 200


def build_blocking_index(source_file: str, source_label: str) -> dict:
    """Build inverted blocking index: key → list of entity_ids."""
    print(f"\n[Index] Building index for {source_label} ...")
    index = defaultdict(set)

    reader = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    total_rows = 0
    for chunk in tqdm(reader, desc=f"  {source_label}"):
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            nn = normalize_name(row.business_name)
            na = normalize_address(row.business_address)
            nc = normalize_country(row.country)
            for key_type, key_val in get_blocking_keys(nn, na, nc):
                full_key = f"{key_type}::{key_val}"
                index[full_key].add(row.entity_id)
        total_rows += len(chunk)
        del chunk

    capped = {k: list(v)[:MAX_CANDIDATES_PER_BLOCK_KEY] for k, v in index.items()}
    print(f"  {source_label}: {total_rows:,} entities, {len(capped):,} unique keys")
    return capped


def generate_test_candidates(
    test_s1_file: str,
    combined_index: dict,
    all_test_s1_ids: set,
    output_file: str,
):
    """Generate candidates for all test S1 entities."""
    print(f"\n[Candidates] Processing {test_s1_file} ...")

    # s1_id → set of candidate IDs
    results = {s1_id: set() for s1_id in all_test_s1_ids}

    reader = pd.read_csv(
        test_s1_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    for chunk in tqdm(reader, desc="  Test S1 chunks"):
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            nn = normalize_name(row.business_name)
            na = normalize_address(row.business_address)
            nc = normalize_country(row.country)

            candidates = set()
            for key_type, key_val in get_blocking_keys(nn, na, nc):
                full_key = f"{key_type}::{key_val}"
                if full_key in combined_index:
                    candidates.update(combined_index[full_key])

            results[s1_id] = candidates
        del chunk

    # Write output (one row per S1 entity, even if no candidates)
    rows = []
    for s1_id in sorted(results.keys()):
        cands = sorted(results[s1_id])
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep="\t", index=False)

    non_empty = df["candidate_entity_ids"].str.len().gt(0).sum()
    total_pairs = df["candidate_entity_ids"].apply(
        lambda x: len(x.split(",")) if x else 0
    ).sum()
    print(f"  {len(df):,} S1 entities, {non_empty:,} with candidates, {total_pairs:,} total pairs")
    print(f"  Saved → {output_file}")


def load_all_test_s1_ids() -> set:
    """Load all test S1 entity IDs."""
    ids = set()
    reader = pd.read_csv(TEST_S1, sep="\t", dtype=str, chunksize=100_000, usecols=["entity_id"])
    for chunk in reader:
        ids.update(chunk["entity_id"].dropna().tolist())
        del chunk
    print(f"  Test S1 entities: {len(ids):,}")
    return ids


def main():
    print("=" * 60)
    print("Test Candidate Generation")
    print("=" * 60)

    # Load all test S1 IDs
    all_test_s1_ids = load_all_test_s1_ids()

    # Build blocking indexes from TEST S2 and S3
    index_s2 = build_blocking_index(TEST_S2, "Test-Source2")
    gc.collect()

    index_s3 = build_blocking_index(TEST_S3, "Test-Source3")
    gc.collect()

    # Merge indexes
    print("\n[Index] Merging S2 + S3 indexes ...")
    combined_index = {}
    all_keys = set(index_s2.keys()) | set(index_s3.keys())
    for k in all_keys:
        lst = index_s2.get(k, []) + index_s3.get(k, [])
        combined_index[k] = lst[:MAX_CANDIDATES_PER_BLOCK_KEY]
    del index_s2, index_s3
    gc.collect()
    print(f"  Combined: {len(combined_index):,} unique keys")

    # Generate candidates
    generate_test_candidates(TEST_S1, combined_index, all_test_s1_ids, OUTPUT_FILE)

    print("\n[Done] Test candidate generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
