"""
05_generate_test_candidates.py
===============================
TWO-STAGE candidate generation for TEST Source 1 entities.

Stage 1 (Blocking): Build inverted index from test S2+S3 using compound blocking keys.
Stage 2 (Pre-filter): Score raw hits with fast token overlap; keep top-K per S1 entity.

Produces: output/test_candidate_pairs.tsv (the file fed directly to the ML model).

Format (validator-compatible):
    source1_entity_id   candidate_entity_ids   (comma-separated)

Every test S1 entity gets a row (empty string if no candidates survive filtering).
"""

import os
import sys
import gc
import csv
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

CHUNK_SIZE = 250_000
MAX_CANDIDATES_PER_BLOCK_KEY = 200
TOP_K_CANDIDATES = 50
MIN_SCORE = 0.10


# ---------------------------------------------------------------------------
# Fast pre-filter scoring
# ---------------------------------------------------------------------------

def token_overlap_score(name_a: str, name_b: str,
                        addr_a: str, addr_b: str,
                        country_a: str, country_b: str) -> float:
    """Fast lightweight similarity score for pre-filtering."""
    if country_a and country_b and country_a != country_b:
        return 0.0

    toks_a = set(t for t in name_a.split() if len(t) >= 3)
    toks_b = set(t for t in name_b.split() if len(t) >= 3)

    if toks_a or toks_b:
        inter = len(toks_a & toks_b)
        union = len(toks_a | toks_b)
        name_jac = inter / union if union else 0.0
    else:
        name_jac = 0.0

    prefix_a = name_a[:4] if name_a else ""
    prefix_b = name_b[:4] if name_b else ""
    prefix_bonus = 0.15 if prefix_a and prefix_a == prefix_b else 0.0

    nums_a = set(re.findall(r"\d+", addr_a))
    nums_b = set(re.findall(r"\d+", addr_b))
    if nums_a and nums_b:
        addr_num_score = len(nums_a & nums_b) / len(nums_a | nums_b)
    else:
        addr_num_score = 0.0

    return 0.6 * name_jac + 0.25 * addr_num_score + prefix_bonus


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def add_to_blocking_index(index: dict, entity_records: dict,
                           source_file: str, source_label: str) -> int:
    """Populate inverted index and entity records store."""
    print(f"\n[Index] Building index for {source_label} ...")
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
            eid = row.entity_id
            nn = normalize_name(row.business_name)
            na = normalize_address(row.business_address)
            nc = normalize_country(row.country)

            entity_records[eid] = (nn, na, nc)

            for key_type, key_val in get_blocking_keys(nn, na, nc):
                full_key = f"{key_type}::{key_val}"
                cands = index.get(full_key)
                if cands is None:
                    index[full_key] = [eid]
                elif len(cands) < MAX_CANDIDATES_PER_BLOCK_KEY:
                    cands.append(eid)

        total_rows += len(chunk)
        del chunk

    print(f"  {source_label}: {total_rows:,} entities (index now: {len(index):,} unique keys)")
    return total_rows


# ---------------------------------------------------------------------------
# Generate + pre-filter
# ---------------------------------------------------------------------------

def generate_test_candidates_filtered(
    test_s1_file: str,
    combined_index: dict,
    entity_records: dict,
    output_file: str,
) -> tuple:
    """Stream filtered candidates to output TSV."""
    print(f"\n[Candidates] Processing {test_s1_file} → {output_file} ...")

    total_entities = 0
    non_empty = 0
    total_pairs = 0
    raw_pairs_total = 0

    reader = pd.read_csv(
        test_s1_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk in tqdm(reader, desc="  Test S1 chunks"):
            chunk = chunk.fillna("")
            for row in chunk.itertuples(index=False):
                s1_id = row.entity_id
                s1_nn = normalize_name(row.business_name)
                s1_na = normalize_address(row.business_address)
                s1_nc = normalize_country(row.country)

                # Stage 1: raw blocking hits
                raw_cands = set()
                for key_type, key_val in get_blocking_keys(s1_nn, s1_na, s1_nc):
                    full_key = f"{key_type}::{key_val}"
                    c_list = combined_index.get(full_key)
                    if c_list:
                        raw_cands.update(c_list)

                raw_pairs_total += len(raw_cands)

                # Stage 2: score + filter
                scored = []
                for cand_id in raw_cands:
                    rec = entity_records.get(cand_id)
                    if rec is None:
                        continue
                    c_nn, c_na, c_nc = rec
                    score = token_overlap_score(s1_nn, c_nn, s1_na, c_na, s1_nc, c_nc)
                    if score >= MIN_SCORE:
                        scored.append((score, cand_id))

                scored.sort(reverse=True)
                top_cands = [cand_id for _, cand_id in scored[:TOP_K_CANDIDATES]]
                cand_str = ",".join(sorted(top_cands))

                writer.writerow([s1_id, cand_str])
                total_entities += 1
                if cand_str:
                    non_empty += 1
                    total_pairs += len(top_cands)

            del chunk

    return total_entities, non_empty, total_pairs, raw_pairs_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Test Candidate Generation [Two-Stage]")
    print(f"  TOP_K_CANDIDATES: {TOP_K_CANDIDATES}")
    print(f"  MIN_SCORE: {MIN_SCORE}")
    print("=" * 60)

    combined_index = {}
    entity_records = {}

    add_to_blocking_index(combined_index, entity_records, TEST_S2, "Test-Source2")
    gc.collect()

    add_to_blocking_index(combined_index, entity_records, TEST_S3, "Test-Source3")
    gc.collect()

    print(f"\n  Final combined index: {len(combined_index):,} unique keys")
    print(f"  Entity records stored: {len(entity_records):,}")

    total_e, non_empty, total_p, raw_total = generate_test_candidates_filtered(
        TEST_S1, combined_index, entity_records, OUTPUT_FILE
    )

    del combined_index, entity_records
    gc.collect()

    reduction = (1 - total_p / max(raw_total, 1)) * 100
    avg_cands = total_p / max(total_e, 1)
    print(f"\n[Stats]")
    print(f"  S1 entities: {total_e:,}")
    print(f"  Raw blocking pairs: {raw_total:,}")
    print(f"  After pre-filter:   {total_p:,}")
    print(f"  Reduction:          {reduction:.1f}%")
    print(f"  Avg candidates/S1:  {avg_cands:.1f}")
    print(f"  With candidates:    {non_empty:,}")
    print("\n[Done] Test candidate generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
