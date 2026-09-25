"""
01_generate_candidates_laptop2.py
==================================
TWO-STAGE candidate generation for S1_Laptop2.tsv (Person 2's portion of Source 1).

Stage 1 (Blocking): Build inverted index from S2+S3 using compound blocking keys.
Stage 2 (Pre-filter): For each S1 entity, score raw blocking hits with fast token
                     overlap + name prefix similarity, then keep only top-K
                     highest-scoring candidates per S1 entity.

This ensures candidate_pairs.tsv is SMALL (fewer candidates = better final ranking)
while maintaining high blocking recall (precision-first filtering with fast heuristics).

Output: output/candidates_laptop2.tsv
Columns: source1_entity_id | candidate_entity_id (one pair per row)
"""

import os
import sys
import gc
import csv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from normalize import normalize_name, normalize_address, normalize_country, get_blocking_keys, get_name_tokens

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

CHUNK_SIZE = 250_000
MAX_CANDIDATES_PER_BLOCK_KEY = 200  # broad blocking cap
TOP_K_CANDIDATES = 50               # keep only top-K after fast pre-filter per S1 entity
MIN_SCORE = 0.10                    # minimum token overlap score to keep a candidate


# ---------------------------------------------------------------------------
# Fast pre-filter scoring
# ---------------------------------------------------------------------------

def token_overlap_score(name_a: str, name_b: str,
                        addr_a: str, addr_b: str,
                        country_a: str, country_b: str) -> float:
    """
    Fast lightweight similarity score for pre-filtering.
    Uses token Jaccard on name + address, country exact match bonus.
    No fuzzy or heavy string ops — must be very fast.
    """
    # Country mismatch → hard reject
    if country_a and country_b and country_a != country_b:
        return 0.0

    # Name token Jaccard
    toks_a = set(t for t in name_a.split() if len(t) >= 3)
    toks_b = set(t for t in name_b.split() if len(t) >= 3)

    if toks_a or toks_b:
        inter = len(toks_a & toks_b)
        union = len(toks_a | toks_b)
        name_jac = inter / union if union else 0.0
    else:
        name_jac = 0.0

    # Quick prefix match bonus (first 4 chars of first token)
    prefix_a = name_a[:4] if name_a else ""
    prefix_b = name_b[:4] if name_b else ""
    prefix_bonus = 0.15 if prefix_a and prefix_a == prefix_b else 0.0

    # Address number overlap
    import re
    nums_a = set(re.findall(r"\d+", addr_a))
    nums_b = set(re.findall(r"\d+", addr_b))
    if nums_a and nums_b:
        addr_num_score = len(nums_a & nums_b) / len(nums_a | nums_b)
    elif not nums_a and not nums_b:
        addr_num_score = 0.0
    else:
        addr_num_score = 0.0

    score = 0.6 * name_jac + 0.25 * addr_num_score + prefix_bonus
    return score


# ---------------------------------------------------------------------------
# Build index: blocking_key -> list of (entity_id, norm_name, norm_addr, norm_country)
# ---------------------------------------------------------------------------

def add_to_blocking_index(index: dict, entity_records: dict,
                          source_file: str, source_label: str) -> int:
    """
    Populate:
      index: key -> list of entity_ids (for fast lookup, capped)
      entity_records: entity_id -> (norm_name, norm_addr, norm_country)
    """
    print(f"\n[Index] Building blocking index for {source_label} ...")
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
        for row in chunk.itertuples(index=False):
            eid = row.entity_id
            nn = normalize_name(row.business_name)
            na = normalize_address(row.business_address)
            nc = normalize_country(row.country)

            # Store normalized record for pre-filter scoring
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

    print(f"  {source_label}: {total_rows:,} entities processed, "
          f"index now has {len(index):,} unique keys")
    return total_rows


# ---------------------------------------------------------------------------
# Generate + pre-filter candidates
# ---------------------------------------------------------------------------

def generate_candidates_filtered(s1_file: str, index: dict,
                                  entity_records: dict, output_writer) -> tuple:
    """
    For each S1 entity:
      1. Look up all blocking hits
      2. Score each hit with fast token overlap
      3. Keep only top-K above MIN_SCORE
      4. Write to output
    """
    reader = pd.read_csv(
        s1_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    total_entities = 0
    total_pairs = 0
    pairs_before_filter = 0

    for chunk in tqdm(reader, desc="  S1 chunks"):
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            s1_nn = normalize_name(row.business_name)
            s1_na = normalize_address(row.business_address)
            s1_nc = normalize_country(row.country)

            # Stage 1: collect raw blocking hits
            raw_cands = set()
            for key_type, key_val in get_blocking_keys(s1_nn, s1_na, s1_nc):
                full_key = f"{key_type}::{key_val}"
                c_list = index.get(full_key)
                if c_list:
                    raw_cands.update(c_list)

            pairs_before_filter += len(raw_cands)

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

            # Keep top-K
            scored.sort(reverse=True)
            top_cands = scored[:TOP_K_CANDIDATES]

            for _, cand_id in top_cands:
                output_writer.writerow({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                })
                total_pairs += 1

            total_entities += 1

        del chunk

    reduction = (1 - total_pairs / max(pairs_before_filter, 1)) * 100
    return total_entities, total_pairs, pairs_before_filter, reduction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Candidate Generation — Laptop 2 (Person 2) [Two-Stage]")
    print(f"  TOP_K_CANDIDATES per S1 entity: {TOP_K_CANDIDATES}")
    print(f"  MIN_SCORE threshold: {MIN_SCORE}")
    print("=" * 60)

    # Build combined inverted index and entity records store
    combined_index = {}
    entity_records = {}  # eid -> (norm_name, norm_addr, norm_country)

    add_to_blocking_index(combined_index, entity_records, SOURCE2, "Source2")
    gc.collect()

    add_to_blocking_index(combined_index, entity_records, SOURCE3, "Source3")
    gc.collect()

    print(f"\n  Final index: {len(combined_index):,} unique blocking keys")
    print(f"  Entity records stored: {len(entity_records):,}")

    # Generate + pre-filter candidates
    print(f"\n[Candidate] Generating + filtering candidates for {S1_LAPTOP2} ...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source1_entity_id", "candidate_entity_id"],
            delimiter="\t",
        )
        writer.writeheader()
        total_entities, total_pairs, raw_pairs, reduction = generate_candidates_filtered(
            S1_LAPTOP2, combined_index, entity_records, writer
        )

    del combined_index, entity_records
    gc.collect()

    avg_cands = total_pairs / max(total_entities, 1)
    print(f"\n[Stats]")
    print(f"  S1 entities processed: {total_entities:,}")
    print(f"  Raw blocking pairs:    {raw_pairs:,}")
    print(f"  After pre-filter:      {total_pairs:,}")
    print(f"  Reduction:             {reduction:.1f}%")
    print(f"  Avg candidates/S1:     {avg_cands:.1f}")
    print(f"\n[Done] Wrote {total_pairs:,} candidate pairs → {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
