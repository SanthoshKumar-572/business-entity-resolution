"""
03_feature_engineering.py
==========================
Compute features for every candidate pair in candidate_pairs.tsv.

Features computed per pair:
  BUSINESS NAME:
    - name_exact_match         : exact normalized name match
    - name_token_overlap       : |intersection| / |union|  (Jaccard on tokens)
    - name_jaccard             : char n-gram Jaccard (3-gram)
    - name_char_sim            : character-level sequence similarity
    - name_prefix_sim          : longest common prefix ratio
    - name_len_diff            : |len(a) - len(b)| / max(len(a), len(b))
    - name_fuzzy_ratio         : RapidFuzz ratio
    - name_fuzzy_partial       : RapidFuzz partial ratio
    - name_fuzzy_token_sort    : RapidFuzz token sort ratio
    - name_fuzzy_token_set     : RapidFuzz token set ratio

  ADDRESS:
    - addr_exact_match         : exact normalized address match
    - addr_token_overlap       : Jaccard on address tokens
    - addr_jaccard             : char n-gram Jaccard (3-gram)
    - addr_char_sim            : character-level sequence similarity
    - addr_numeric_overlap     : Jaccard on numeric substrings
    - addr_prefix_sim          : longest common prefix ratio
    - addr_len_diff            : length difference ratio
    - addr_fuzzy_ratio         : RapidFuzz ratio

  COUNTRY:
    - country_exact_match      : exact raw country match
    - country_norm_match       : exact normalized country match

  COMBINED:
    - name_addr_combined       : weighted avg of name + addr similarity
    - is_name_missing_s1       : S1 name is empty
    - is_name_missing_cand     : Candidate name is empty
    - is_addr_missing_s1       : S1 address is empty
    - is_addr_missing_cand     : Candidate address is empty

Output: output/features_train.parquet / output/features_test.parquet
"""

import os
import sys
import gc
from difflib import SequenceMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from normalize import (
    normalize_name, normalize_address, normalize_country,
    get_name_tokens, get_address_tokens, extract_numbers,
)

try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("[WARN] rapidfuzz not available — fuzzy features will be 0")

OUTPUT_DIR = os.path.join(ROOT, "output")
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")

CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

OUTPUT_TRAIN_FEATURES = os.path.join(OUTPUT_DIR, "features_train.parquet")
OUTPUT_TEST_FEATURES = os.path.join(OUTPUT_DIR, "features_test.parquet")

CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Helper feature functions
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 3) -> set:
    """Character n-grams of a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def token_jaccard(a_tokens: set, b_tokens: set) -> float:
    """Jaccard on token sets."""
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def numeric_jaccard(a: str, b: str) -> float:
    """Jaccard on numeric substrings."""
    a_nums = set(extract_numbers(a))
    b_nums = set(extract_numbers(b))
    if not a_nums and not b_nums:
        return 1.0
    if not a_nums or not b_nums:
        return 0.0
    inter = len(a_nums & b_nums)
    union = len(a_nums | b_nums)
    return inter / union if union else 0.0


def prefix_sim(a: str, b: str) -> float:
    """Longest common prefix ratio."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    common = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            common += 1
        else:
            break
    return common / max_len


def len_diff_ratio(a: str, b: str) -> float:
    """Length difference ratio."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    return abs(la - lb) / max(la, lb, 1)


def char_sim(a: str, b: str) -> float:
    """Character-level sequence similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_features(a: str, b: str) -> tuple:
    """RapidFuzz features: ratio, partial_ratio, token_sort, token_set."""
    if not HAS_RAPIDFUZZ:
        return (0.0, 0.0, 0.0, 0.0)
    if not a and not b:
        return (1.0, 1.0, 1.0, 1.0)
    if not a or not b:
        return (0.0, 0.0, 0.0, 0.0)
    r = rfuzz.ratio(a, b)
    pr = rfuzz.partial_ratio(a, b)
    ts = rfuzz.token_sort_ratio(a, b)
    tset = rfuzz.token_set_ratio(a, b)
    return (r / 100.0, pr / 100.0, ts / 100.0, tset / 100.0)


# ---------------------------------------------------------------------------
# Entity lookup: pre-load ONLY needed normalized records into memory
# ---------------------------------------------------------------------------

def collect_needed_ids(candidate_file: str) -> tuple:
    """Collect unique S1 IDs and Candidate IDs present in candidate_file."""
    print(f"  Scanning candidate file {candidate_file} for active entity IDs ...")
    needed_s1 = set()
    needed_cands = set()
    if not os.path.isfile(candidate_file):
        return needed_s1, needed_cands

    reader = pd.read_csv(candidate_file, sep="\t", dtype=str, chunksize=100_000)
    for chunk in reader:
        chunk = chunk.fillna("")
        cols = chunk.columns.tolist()
        s1_col = "source1_entity_id" if "source1_entity_id" in cols else cols[0]
        c_col = "candidate_entity_ids" if "candidate_entity_ids" in cols else (cols[1] if len(cols) > 1 else "")
        for row in chunk.itertuples(index=False):
            s1_id = str(getattr(row, s1_col, "")).strip()
            c_str = str(getattr(row, c_col, "")).strip() if c_col else ""
            if s1_id and c_str:
                needed_s1.add(s1_id)
                for cid in c_str.split(","):
                    cid = cid.strip()
                    if cid:
                        needed_cands.add(cid)
        del chunk

    print(f"  Active entities required: {len(needed_s1):,} S1 IDs, {len(needed_cands):,} candidate IDs")
    return needed_s1, needed_cands


def load_entity_lookup(source_files: list, needed_ids: set = None) -> dict:
    """
    Load entity records from source files filtered by needed_ids.
    Returns dict: entity_id -> (norm_name, norm_addr, norm_country, raw_country).
    """
    lookup = {}
    for fpath in source_files:
        if not os.path.isfile(fpath):
            print(f"  [WARN] Missing: {fpath}")
            continue
        print(f"  Loading {os.path.basename(fpath)} ...")
        reader = pd.read_csv(
            fpath, sep="\t", dtype=str, chunksize=250_000,
            usecols=["entity_id", "business_name", "business_address", "country"],
        )
        for chunk in tqdm(reader, desc=f"    {os.path.basename(fpath)}"):
            chunk = chunk.fillna("")
            for row in chunk.itertuples(index=False):
                eid = row.entity_id
                if needed_ids is not None and eid not in needed_ids:
                    continue
                lookup[eid] = (
                    normalize_name(row.business_name),
                    normalize_address(row.business_address),
                    normalize_country(row.country),
                    row.country,
                )
            del chunk
        gc.collect()
    print(f"  Loaded entity lookup size: {len(lookup):,}")
    return lookup


def load_ground_truth() -> dict:
    """Load ground truth: {s1_id: set(matched_entity_ids)}."""
    print(f"  Loading ground truth from {GROUND_TRUTH} ...")
    gt = {}
    df = pd.read_csv(GROUND_TRUTH, sep="\t", dtype=str).fillna("")
    for row in df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids
        if matched.strip():
            gt[s1_id] = set(matched.split(","))
        else:
            gt[s1_id] = set()
    print(f"  Ground truth: {len(gt):,} S1 entities")
    return gt


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features_for_pair(
    s1_id: str,
    cand_id: str,
    s1_lookup: dict,
    cand_lookup: dict,
    ground_truth: dict,
) -> dict:
    """Compute all features for one candidate pair."""

    s1_data = s1_lookup.get(s1_id)
    cand_data = cand_lookup.get(cand_id)

    if s1_data is None:
        s1_nn, s1_na, s1_nc, s1_rc = "", "", "", ""
    else:
        s1_nn, s1_na, s1_nc, s1_rc = s1_data

    if cand_data is None:
        c_nn, c_na, c_nc, c_rc = "", "", "", ""
    else:
        c_nn, c_na, c_nc, c_rc = cand_data

    # Token sets
    s1_name_tokens = get_name_tokens(s1_nn)
    c_name_tokens = get_name_tokens(c_nn)
    s1_addr_tokens = get_address_tokens(s1_na)
    c_addr_tokens = get_address_tokens(c_na)

    # Fuzzy name features
    fn_ratio, fn_partial, fn_token_sort, fn_token_set = fuzzy_features(s1_nn, c_nn)
    fa_ratio, _, _, _ = fuzzy_features(s1_na, c_na)

    # Sequence similarities
    name_cs = char_sim(s1_nn, c_nn)
    addr_cs = char_sim(s1_na, c_na)

    combined_sim = 0.6 * fn_ratio + 0.4 * fa_ratio

    feat = {
        "source1_entity_id": s1_id,
        "candidate_entity_id": cand_id,
        # Name features
        "name_exact_match": int(s1_nn == c_nn and s1_nn != ""),
        "name_token_overlap": token_jaccard(s1_name_tokens, c_name_tokens),
        "name_jaccard": jaccard_ngram(s1_nn, c_nn, 3),
        "name_char_sim": name_cs,
        "name_prefix_sim": prefix_sim(s1_nn, c_nn),
        "name_len_diff": len_diff_ratio(s1_nn, c_nn),
        "name_fuzzy_ratio": fn_ratio,
        "name_fuzzy_partial": fn_partial,
        "name_fuzzy_token_sort": fn_token_sort,
        "name_fuzzy_token_set": fn_token_set,
        # Address features
        "addr_exact_match": int(s1_na == c_na and s1_na != ""),
        "addr_token_overlap": token_jaccard(s1_addr_tokens, c_addr_tokens),
        "addr_jaccard": jaccard_ngram(s1_na, c_na, 3),
        "addr_char_sim": addr_cs,
        "addr_numeric_overlap": numeric_jaccard(s1_na, c_na),
        "addr_prefix_sim": prefix_sim(s1_na, c_na),
        "addr_len_diff": len_diff_ratio(s1_na, c_na),
        "addr_fuzzy_ratio": fa_ratio,
        # Country features
        "country_exact_match": int(s1_rc.strip() == c_rc.strip() and s1_rc.strip() != ""),
        "country_norm_match": int(s1_nc == c_nc and s1_nc != ""),
        # Combined
        "name_addr_combined": combined_sim,
        # Missing indicators
        "is_name_missing_s1": int(s1_nn == ""),
        "is_name_missing_cand": int(c_nn == ""),
        "is_addr_missing_s1": int(s1_na == ""),
        "is_addr_missing_cand": int(c_na == ""),
        # Source indicator
        "is_s2_candidate": int(cand_id.startswith("S2-")),
        "is_s3_candidate": int(cand_id.startswith("S3-")),
    }

    # Label
    if ground_truth is not None:
        gt_matches = ground_truth.get(s1_id, set())
        feat["label"] = int(cand_id in gt_matches)
    else:
        feat["label"] = -1

    return feat


def process_candidate_file(
    candidate_file: str,
    s1_lookup: dict,
    cand_lookup: dict,
    ground_truth: dict,
    output_path: str,
    mode: str = "train",
):
    """Process all candidate pairs and save features incrementally using ParquetWriter."""
    print(f"\n[Features] Processing {candidate_file} → {output_path}")

    if not os.path.isfile(candidate_file):
        print(f"  ERROR: {candidate_file} not found.")
        return

    writer = None
    all_features = []
    total_written = 0

    reader = pd.read_csv(
        candidate_file,
        sep="\t",
        dtype=str,
        chunksize=5_000,
    )

    for chunk in tqdm(reader, desc="  Candidate chunks"):
        chunk = chunk.fillna("")
        cols = chunk.columns.tolist()
        s1_col = "source1_entity_id" if "source1_entity_id" in cols else cols[0]
        c_col = "candidate_entity_ids" if "candidate_entity_ids" in cols else (cols[1] if len(cols) > 1 else "")

        for row in chunk.itertuples(index=False):
            s1_id = str(getattr(row, s1_col, "")).strip()
            cand_str = str(getattr(row, c_col, "")).strip() if c_col else ""
            if not cand_str:
                continue

            for cand_id in cand_str.split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    feat = compute_features_for_pair(
                        s1_id, cand_id, s1_lookup, cand_lookup, ground_truth
                    )
                    all_features.append(feat)

        if len(all_features) >= 100_000:
            df_chunk = pd.DataFrame(all_features)
            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_written += len(df_chunk)
            all_features = []
            del df_chunk, table
            gc.collect()

        del chunk

    if all_features:
        df_chunk = pd.DataFrame(all_features)
        table = pa.Table.from_pandas(df_chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_written += len(df_chunk)
        del df_chunk, table

    if writer is not None:
        writer.close()

    print(f"  Saved {total_written:,} feature rows → {output_path}")


def main():
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    print(f"  Mode: {mode}")

    if mode == "train":
        s1_source = os.path.join(TRAIN_DIR, "train_source1.tsv")
        s2_source = os.path.join(TRAIN_DIR, "train_source2.tsv")
        s3_source = os.path.join(TRAIN_DIR, "train_source3.tsv")
        output_path = OUTPUT_TRAIN_FEATURES
        candidate_file = CANDIDATE_FILE
        ground_truth = load_ground_truth()
    else:
        s1_source = os.path.join(TEST_DIR, "test_source1.tsv")
        s2_source = os.path.join(TEST_DIR, "test_source2.tsv")
        s3_source = os.path.join(TEST_DIR, "test_source3.tsv")
        output_path = OUTPUT_TEST_FEATURES
        candidate_file = os.path.join(OUTPUT_DIR, "test_candidate_pairs.tsv")
        ground_truth = None

    # Step 1: Scan candidate file to collect active entity IDs
    needed_s1, needed_cands = collect_needed_ids(candidate_file)

    # Step 2: Load only required entity records
    print("\n[Lookup] Loading Source 1 records ...")
    s1_lookup = load_entity_lookup([s1_source], needed_ids=needed_s1)

    print("\n[Lookup] Loading Source 2 + Source 3 records ...")
    cand_lookup = load_entity_lookup([s2_source, s3_source], needed_ids=needed_cands)

    # Step 3: Compute features and write incrementally
    process_candidate_file(
        candidate_file=candidate_file,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
        ground_truth=ground_truth,
        output_path=output_path,
        mode=mode,
    )

    print("\n[Done] Feature engineering complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
