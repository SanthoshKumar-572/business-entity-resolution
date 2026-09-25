"""
01_generate_candidates_laptop2.py
==================================
Candidate generation for S1_Laptop2.tsv (Person 2's portion of Source 1).

This script generates output/candidates_laptop2.tsv containing:
    source1_entity_id   candidate_entity_id

Strategy:
  - Build inverted blocking indexes from Source 2 and Source 3 (chunked)
  - For each S1_Laptop2 entity, look up its blocking keys in the indexes
  - Emit candidate pairs without doing pairwise comparison

Memory-efficient: processes S2/S3 in chunks, builds dict-based indexes.
Performance: uses itertuples() (~10x faster than iterrows).
"""

import os
import sys
import gc
import csv
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # student_resource/
sys.path.insert(0, SCRIPT_DIR)

from normalize import normalize_name, normalize_address, normalize_country, get_blocking_keys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_LAPTOP2 = os.path.join(TRAIN_DIR, "S1_Laptop2.tsv")
SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidates_laptop2.tsv")

CHUNK_SIZE = 200_000
MAX_CANDIDATES_PER_BLOCK_KEY = 200   # hard cap per blocking key to avoid cartesian explosion


def build_blocking_index(source_file: str, source_label: str) -> dict:
    """
    Build an inverted index: blocking_key → set of entity_ids.

    Processes the source file in chunks to stay within memory limits.
    Uses itertuples() (not iterrows) for fast row access.
    Returns dict[str, list[str]].
    """
    print(f"\n[Index] Building blocking index for {source_label} ...")
    index = defaultdict(set)

    reader = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    total_rows = 0
    for chunk in tqdm(reader, desc=f"  {source_label} chunks"):
        chunk = chunk.fillna("")
        # itertuples is ~10x faster than iterrows for large DataFrames
        for row in chunk.itertuples(index=False):
            norm_name = normalize_name(row.business_name)
            norm_addr = normalize_address(row.business_address)
            norm_ctry = normalize_country(row.country)

            for key_type, key_val in get_blocking_keys(norm_name, norm_addr, norm_ctry):
                full_key = f"{key_type}::{key_val}"
                index[full_key].add(row.entity_id)

        total_rows += len(chunk)
        del chunk

    # Convert sets to lists, apply cap
    capped = {k: list(v)[:MAX_CANDIDATES_PER_BLOCK_KEY] for k, v in index.items()}

    print(f"  {source_label}: indexed {total_rows:,} entities, {len(capped):,} unique blocking keys")
    return capped


def generate_candidates(s1_file: str, index: dict, output_writer, written_pairs: set) -> int:
    """
    For each S1 entity in s1_file, look up blocking keys in the combined index
    and write candidate pairs. Uses itertuples() for speed.
    """
    reader = pd.read_csv(
        s1_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    total_pairs = 0
    for chunk in tqdm(reader, desc="  S1 chunks"):
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            norm_name = normalize_name(row.business_name)
            norm_addr = normalize_address(row.business_address)
            norm_ctry = normalize_country(row.country)

            candidates = set()
            for key_type, key_val in get_blocking_keys(norm_name, norm_addr, norm_ctry):
                full_key = f"{key_type}::{key_val}"
                if full_key in index:
                    candidates.update(index[full_key])

            for cand_id in candidates:
                pair = (s1_id, cand_id)
                if pair not in written_pairs:
                    written_pairs.add(pair)
                    output_writer.writerow({"source1_entity_id": s1_id, "candidate_entity_id": cand_id})
                    total_pairs += 1

        del chunk

    return total_pairs


def main():
    print("=" * 60)
    print("Candidate Generation — Laptop 2 (Person 2)")
    print("=" * 60)

    # Step 1: Build indexes for S2 and S3
    index_s2 = build_blocking_index(SOURCE2, "Source2")
    gc.collect()

    index_s3 = build_blocking_index(SOURCE3, "Source3")
    gc.collect()

    # Step 2: Merge indexes
    print("\n[Index] Merging S2 + S3 indexes ...")
    combined_index = {}
    all_keys = set(index_s2.keys()) | set(index_s3.keys())
    for k in all_keys:
        lst = index_s2.get(k, []) + index_s3.get(k, [])
        combined_index[k] = lst[:MAX_CANDIDATES_PER_BLOCK_KEY]
    del index_s2, index_s3
    gc.collect()
    print(f"  Combined index: {len(combined_index):,} unique keys")

    # Step 3: Generate candidates for S1_Laptop2
    print(f"\n[Candidate] Generating candidates for {S1_LAPTOP2} ...")
    written_pairs = set()
    total_pairs = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source1_entity_id", "candidate_entity_id"],
            delimiter="\t",
        )
        writer.writeheader()
        total_pairs = generate_candidates(S1_LAPTOP2, combined_index, writer, written_pairs)

    print(f"\n[Done] Wrote {total_pairs:,} candidate pairs → {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
