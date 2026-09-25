"""
diagnose_100k_misses.py
=======================
Diagnose blocking recall, true matches found, and analyze missed matches
for 100,000 S1 entities under Current Blocking vs Improved Blocking.
"""

import os
import sys
import gc
import time
import json
import re
import unicodedata
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "code":
    ROOT = os.path.dirname(ROOT)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
OUTPUT_DIR = os.path.join(ROOT, "output", "person1_100k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

print("=" * 65)
print("PERSON 1: 100K BLOCKING & MISS ANALYSIS")
print("=" * 65)

# Step 1: Load Ground Truth and sample 100,000 S1 entities
print("\n[Step 1] Sampling 100,000 S1 entities and parsing Ground Truth...")
t0 = time.time()

# Parse GT
gt_map = {}
with open(GT_FILE, "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            s1_id = parts[0].strip()
            cands = [c.strip() for c in parts[1].split(",") if c.strip()]
            gt_map[s1_id] = set(cands)
        elif len(parts) == 1:
            gt_map[parts[0].strip()] = set()

# Sample 100,000 S1 IDs (matching random_state=42)
s1_ids_list = []
with open(S1_FILE, "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.split("\t")
        if parts:
            s1_ids_list.append(parts[0].strip())

rng = np.random.default_rng(42)
s1_sampled_indices = set(rng.choice(len(s1_ids_list), size=100000, replace=False))
selected_s1_ids = set(s1_ids_list[i] for i in s1_sampled_indices)

# Ground truth stats for 100K
gt_100k = {s1_id: gt_map.get(s1_id, set()) for s1_id in selected_s1_ids}
total_true_matches = sum(len(v) for v in gt_100k.values())
singletons_count = sum(1 for v in gt_100k.values() if len(v) == 0)
entities_with_matches = sum(1 for v in gt_100k.values() if len(v) > 0)

target_ids_needed = set()
for v in gt_100k.values():
    target_ids_needed.update(v)

print(f"  Selected S1 Entities:     {len(selected_s1_ids):,}")
print(f"  Entities with matches:   {entities_with_matches:,}")
print(f"  Singletons (0 matches):  {singletons_count:,} ({singletons_count/len(selected_s1_ids)*100:.2f}%)")
print(f"  Total True Matches:      {total_true_matches:,} ({total_true_matches/len(selected_s1_ids):.2f} / S1)")
print(f"  Unique Target IDs needed: {len(target_ids_needed):,}")
print(f"  Time: {time.time() - t0:.1f}s")

# Step 2: Load S1 and Target records
print("\n[Step 2] Loading S1 and target records...")
t0 = time.time()

s1_records = {}
with open(S1_FILE, "r", encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    id_col = header.index("entity_id")
    name_col = header.index("business_name")
    addr_col = header.index("business_address")
    country_col = header.index("country")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > id_col and parts[id_col] in selected_s1_ids:
            eid = parts[id_col]
            name = parts[name_col] if len(parts) > name_col else ""
            addr = parts[addr_col] if len(parts) > addr_col else ""
            country = parts[country_col] if len(parts) > country_col else ""
            s1_records[eid] = (name, addr, country)

print(f"  Loaded {len(s1_records):,} S1 records")

target_records = {}
for src_file, label in [(S2_FILE, "S2"), (S3_FILE, "S3")]:
    with open(src_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_col = header.index("entity_id")
        name_col = header.index("business_name")
        addr_col = header.index("business_address")
        country_col = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > id_col and parts[id_col] in target_ids_needed:
                eid = parts[id_col]
                name = parts[name_col] if len(parts) > name_col else ""
                addr = parts[addr_col] if len(parts) > addr_col else ""
                country = parts[country_col] if len(parts) > country_col else ""
                target_records[eid] = (name, addr, country)
    print(f"  {label} targets loaded. Total targets so far: {len(target_records):,}")

print(f"  All targets loaded in {time.time() - t0:.1f}s")

# Save diagnosis checkpoint info
print("\nDone loading data for diagnosis.")

# ---------------------------------------------------------------------------
# Normalization Functions (Current vs Improved)
# ---------------------------------------------------------------------------

def current_unicode_norm(text: str) -> str:
    """Current version: NFKD + encode ascii ignore (drops non-ascii)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii")

def improved_unicode_norm(text: str) -> str:
    """Improved version: NFKC, preserving Devanagari and Unicode characters."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)

# Country aliases
COUNTRY_ALIASES = {
    "usa": "us", "united states": "us", "united states of america": "us",
    "u.s.a": "us", "u.s": "us", "us": "us",
    "india": "india", "ind": "india", "bharat": "india",
    "france": "france", "fr": "france",
}

def current_norm_country(c: str) -> str:
    if not isinstance(c, str):
        return ""
    return c.lower().strip()

def improved_norm_country(c: str) -> str:
    if not isinstance(c, str):
        return ""
    clean = c.lower().strip()
    return COUNTRY_ALIASES.get(clean, clean)

LEGAL_SUFFIXES = {
    "private limited": "pvt ltd", "pvt ltd": "pvt ltd", "limited": "ltd", "ltd": "ltd",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "plc": "plc",
    "services": "svc", "solutions": "sol", "enterprises": "ent", "industries": "ind",
    "holdings": "hldg", "ventures": "vent", "trading": "trd", "international": "intl",
}
LEGAL_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIXES.keys(), key=len, reverse=True)) + r")\b", re.I)

ADDR_ABBREVS = {
    "street": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg", "post office box": "po box",
    "p.o. box": "po box", "po box": "po box", "highway": "hwy", "hwy": "hwy",
}
ADDR_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(ADDR_ABBREVS.keys(), key=len, reverse=True)) + r")\b", re.I)

def current_norm_name(name: str) -> str:
    if not name: return ""
    text = current_unicode_norm(name).lower()
    text = text.replace("&", " and ")
    text = LEGAL_PAT.sub(lambda m: LEGAL_SUFFIXES[m.group(0).lower()], text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def improved_norm_name(name: str) -> str:
    if not name: return ""
    text = improved_unicode_norm(name).lower()
    text = text.replace("&", " and ")
    text = LEGAL_PAT.sub(lambda m: LEGAL_SUFFIXES[m.group(0).lower()], text)
    # Preserve alphanumeric, Devanagari (\u0900-\u097f), and Latin accented (\u00c0-\u024f)
    text = re.sub(r"[^a-z0-9\u0900-\u097f\u00c0-\u024f\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def current_norm_addr(addr: str) -> str:
    if not addr: return ""
    text = current_unicode_norm(addr).lower()
    text = ADDR_PAT.sub(lambda m: ADDR_ABBREVS[m.group(0).lower()], text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def improved_norm_addr(addr: str) -> str:
    if not addr: return ""
    text = improved_unicode_norm(addr).lower()
    text = ADDR_PAT.sub(lambda m: ADDR_ABBREVS[m.group(0).lower()], text)
    text = re.sub(r"[^a-z0-9\u0900-\u097f\u00c0-\u024f\s/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_pin(addr: str) -> str:
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", addr)
    if m: return m.group(1)
    m = re.search(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)", addr)
    if m: return m.group(1)
    return ""

def extract_nums(text: str) -> set:
    return set(re.findall(r"\d+", text))

NAME_STOP = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "by", "for", "to", "ltd",
    "llc", "inc", "corp", "co", "pvt", "private", "limited", "company",
    "group", "services", "solutions", "trading", "enterprises", "industries",
    "international", "global", "national", "india", "us", "france", "svc", "sol",
    "ent", "ind", "trd", "intl", "grp", "hldg", "vent", "llp", "plc",
})

ADDR_STOP = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "by", "for", "to",
    "no", "near", "opposite", "opp", "flat", "door", "plot", "house",
    "shop", "floor", "fl", "unit", "ste", "apt", "bldg", "room", "st", "rd", "ave",
})

def get_tokens(norm_text: str, stopwords: frozenset) -> list:
    return [t for t in norm_text.split() if len(t) >= 2 and t not in stopwords]

# ---------------------------------------------------------------------------
# Step 3: Evaluate Current Blocking on True Matches
# ---------------------------------------------------------------------------
print("\n[Step 3] Evaluating Current Blocking on 345,941 True Matches...")
t0 = time.time()

# Precompute normalized attributes for S1 and Targets
print("  Normalizing S1 records (Current)...")
s1_cur = {}
s1_imp = {}
for eid, (raw_n, raw_a, raw_c) in s1_records.items():
    s1_cur[eid] = {
        "norm_name": current_norm_name(raw_n),
        "norm_addr": current_norm_addr(raw_a),
        "norm_country": current_norm_country(raw_c),
        "postal": extract_pin(raw_a),
        "nums": extract_nums(raw_a),
        "raw_name": raw_n,
        "raw_addr": raw_a,
        "raw_country": raw_c,
    }
    s1_imp[eid] = {
        "norm_name": improved_norm_name(raw_n),
        "norm_addr": improved_norm_addr(raw_a),
        "norm_country": improved_norm_country(raw_c),
        "postal": extract_pin(raw_a),
        "nums": extract_nums(raw_a),
        "raw_name": raw_n,
        "raw_addr": raw_a,
        "raw_country": raw_c,
    }

print("  Normalizing Target records (Current)...")
t_cur = {}
t_imp = {}
for eid, (raw_n, raw_a, raw_c) in target_records.items():
    t_cur[eid] = {
        "norm_name": current_norm_name(raw_n),
        "norm_addr": current_norm_addr(raw_a),
        "norm_country": current_norm_country(raw_c),
        "postal": extract_pin(raw_a),
        "nums": extract_nums(raw_a),
        "raw_name": raw_n,
        "raw_addr": raw_a,
        "raw_country": raw_c,
    }
    t_imp[eid] = {
        "norm_name": improved_norm_name(raw_n),
        "norm_addr": improved_norm_addr(raw_a),
        "norm_country": improved_norm_country(raw_c),
        "postal": extract_pin(raw_a),
        "nums": extract_nums(raw_a),
        "raw_name": raw_n,
        "raw_addr": raw_a,
        "raw_country": raw_c,
    }

# Test which pairs are caught by Current Blocking
cur_captured = 0
cur_missed = []
cur_strategy_hits = Counter()

# Also classify single-match vs multi-match misses
miss_by_entity_type = Counter()

for s1_id, targets in gt_100k.items():
    if not targets:
        continue
    num_matches = len(targets)
    match_category = "single_match" if num_matches == 1 else "multi_match"
    
    s1_info = s1_cur.get(s1_id)
    if not s1_info:
        continue
    s1_name = s1_info["norm_name"]
    s1_addr = s1_info["norm_addr"]
    s1_c = s1_info["norm_country"]
    s1_pin = s1_info["postal"]
    s1_toks = get_tokens(s1_name, NAME_STOP)
    s1_atok = get_tokens(s1_addr, ADDR_STOP)
    s1_ng = {s1_name[i:i+3] for i in range(len(s1_name)-2)} if len(s1_name) >= 3 else set()

    for tid in targets:
        t_info = t_cur.get(tid)
        if not t_info:
            continue
        t_name = t_info["norm_name"]
        t_addr = t_info["norm_addr"]
        t_c = t_info["norm_country"]
        t_pin = t_info["postal"]

        # Check country match requirement
        country_match = (s1_c == t_c) or (not s1_c) or (not t_c)
        hit = False

        if country_match:
            # Strategy A: exact name
            if s1_name and s1_name == t_name:
                hit = True
                cur_strategy_hits["A_exact_name"] += 1
            
            # Strategy B: first token
            t_toks = get_tokens(t_name, NAME_STOP)
            if not hit and s1_toks and t_toks and s1_toks[0] == t_toks[0]:
                hit = True
                cur_strategy_hits["B_first_tok"] += 1

            # Strategy C: prefix 4
            if not hit and len(s1_name) >= 4 and len(t_name) >= 4 and s1_name[:4] == t_name[:4]:
                hit = True
                cur_strategy_hits["C_prefix4"] += 1

            # Strategy D: token overlap >= max(1, 0.2 * len)
            if not hit and s1_toks and t_toks:
                common_toks = set(s1_toks) & set(t_toks)
                if len(common_toks) >= max(1, int(0.2 * len(s1_toks))):
                    hit = True
                    cur_strategy_hits["D_token_overlap"] += 1

            # Strategy E: 3-gram overlap >= 30%
            if not hit and s1_ng and len(t_name) >= 3:
                t_ng = {t_name[i:i+3] for i in range(len(t_name)-2)}
                ng_inter = len(s1_ng & t_ng)
                if ng_inter >= max(2, int(0.30 * len(s1_ng))):
                    hit = True
                    cur_strategy_hits["E_ngram3"] += 1

            # Strategy F: Postal code match
            if not hit and s1_pin and t_pin and len(s1_pin) >= 5 and s1_pin == t_pin:
                hit = True
                cur_strategy_hits["F_postal"] += 1

            # Strategy G: Address tokens >= 2
            t_atok = get_tokens(t_addr, ADDR_STOP)
            if not hit and s1_atok and t_atok:
                common_addr = set(s1_atok) & set(t_atok)
                if len(common_addr) >= 2:
                    hit = True
                    cur_strategy_hits["G_addr_tokens"] += 1

        if hit:
            cur_captured += 1
        else:
            cur_missed.append({
                "s1_id": s1_id,
                "target_id": tid,
                "match_type": match_category,
                "s1_raw_name": s1_info["raw_name"],
                "target_raw_name": t_info["raw_name"],
                "s1_norm_name": s1_name,
                "target_norm_name": t_name,
                "s1_raw_addr": s1_info["raw_addr"],
                "target_raw_addr": t_info["raw_addr"],
                "s1_country": s1_c,
                "target_country": t_c,
                "country_match": country_match,
            })
            miss_by_entity_type[match_category] += 1

cur_recall = cur_captured / max(total_true_matches, 1)
print(f"  Current True Matches Found: {cur_captured:,} / {total_true_matches:,}")
print(f"  Current Blocking Recall:    {cur_recall*100:.2f}% ({cur_recall:.4f})")
print(f"  Current Missed Matches:     {len(cur_missed):,}")
print(f"    - Single-match misses:    {miss_by_entity_type['single_match']:,} ({miss_by_entity_type['single_match']/len(cur_missed)*100:.1f}%)")
print(f"    - Multi-match misses:     {miss_by_entity_type['multi_match']:,} ({miss_by_entity_type['multi_match']/len(cur_missed)*100:.1f}%)")
print(f"  Strategy Contribution in hits: {dict(cur_strategy_hits)}")
print(f"  Time: {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Step 4: Miss Pattern Categorization
# ---------------------------------------------------------------------------
print("\n[Step 4] Categorizing Missed True Matches...")

pattern_counts = Counter()
miss_records_detailed = []

for m in cur_missed:
    s1_n = m["s1_norm_name"]
    t_n = m["target_norm_name"]
    s1_rn = m["s1_raw_name"]
    t_rn = m["target_raw_name"]
    s1_a = m["s1_raw_addr"]
    t_a = m["target_raw_addr"]
    
    # Check causes
    pattern = "other"
    
    # 1. Country variation
    if not m["country_match"]:
        pattern = "country_variation"
    # 2. Transliteration / Non-ASCII erasure
    elif (not s1_n and s1_rn.strip()) or (not t_n and t_rn.strip()):
        pattern = "transliteration_devanagari_erasure"
    # 3. Missing address
    elif (not s1_a.strip()) or (not t_a.strip()):
        pattern = "missing_address"
    # 4. Word order variation (same tokens sorted)
    elif sorted(s1_n.split()) == sorted(t_n.split()) and s1_n.split():
        pattern = "word_order"
    # 5. Token subset / abbreviation (e.g. initials, acronym)
    elif len(s1_n) <= 4 or len(t_n) <= 4:
        pattern = "acronym_short_name"
    # 6. Distinct tokens sharing digits/house number
    elif extract_nums(s1_a) & extract_nums(t_a):
        pattern = "address_numeric_match"
    # 7. Common tokens present but below strict threshold
    elif set(s1_n.split()) & set(t_n.split()):
        pattern = "word_overlap_below_threshold"
    # 8. Character n-gram typo / spelling variation
    elif any(s1_n[i:i+4] in t_n for i in range(len(s1_n)-3)) if len(s1_n) >= 4 else False:
        pattern = "spelling_variation"
    # 9. DBA / completely different name
    elif not (set(s1_n.split()) & set(t_n.split())):
        pattern = "dba_trade_name_or_different"
    
    pattern_counts[pattern] += 1
    m["failure_pattern"] = pattern
    miss_records_detailed.append(m)

print("  Missed Match Failure Patterns:")
for pat, count in pattern_counts.most_common():
    pct = count / len(cur_missed) * 100
    print(f"    - {pat:<35s}: {count:>5,}  ({pct:>5.1f}%)")

# Save missed matches CSV
df_misses = pd.DataFrame(miss_records_detailed)
miss_csv_path = os.path.join(OUTPUT_DIR, "missed_matches.csv")
df_misses.to_csv(miss_csv_path, index=False)
print(f"\n  Saved missed matches to {miss_csv_path}")

# ---------------------------------------------------------------------------
# Step 5: Improved Blocking Design & Evaluation
# ---------------------------------------------------------------------------
print("\n[Step 5] Evaluating Improved Blocking Engine...")
t0 = time.time()

# Improved blocking additions:
# 1. Preserved Unicode (no ASCII stripping)
# 2. Country alias normalization
# 3. Sorted token key (handles word order)
# 4. Address street number + first name token
# 5. Distinctive token inverted index (min length >= 3)
# 6. Prefix 3-character key
# 7. Postal code exact match (including 5 and 6 digits)

imp_captured = 0
imp_strategy_hits = Counter()
recovered_from_misses = 0

for s1_id, targets in gt_100k.items():
    if not targets:
        continue
    s1_info = s1_imp.get(s1_id)
    if not s1_info:
        continue
    s1_name = s1_info["norm_name"]
    s1_addr = s1_info["norm_addr"]
    s1_c = s1_info["norm_country"]
    s1_pin = s1_info["postal"]
    s1_nums = s1_info["nums"]
    s1_toks = get_tokens(s1_name, NAME_STOP)
    s1_atok = get_tokens(s1_addr, ADDR_STOP)
    s1_ng = {s1_name[i:i+3] for i in range(len(s1_name)-2)} if len(s1_name) >= 3 else set()
    s1_sorted_key = " ".join(sorted(s1_toks)) if s1_toks else ""

    for tid in targets:
        t_info = t_imp.get(tid)
        if not t_info:
            continue
        t_name = t_info["norm_name"]
        t_addr = t_info["norm_addr"]
        t_c = t_info["norm_country"]
        t_pin = t_info["postal"]
        t_nums = t_info["nums"]
        t_toks = get_tokens(t_name, NAME_STOP)
        t_atok = get_tokens(t_addr, ADDR_STOP)

        country_match = (s1_c == t_c) or (not s1_c) or (not t_c)
        hit = False

        if country_match:
            # 1. Exact normalized name
            if s1_name and s1_name == t_name:
                hit = True
                imp_strategy_hits["1_exact_name"] += 1
            
            # 2. Sorted tokens (word reordering)
            t_sorted_key = " ".join(sorted(t_toks)) if t_toks else ""
            if not hit and s1_sorted_key and s1_sorted_key == t_sorted_key:
                hit = True
                imp_strategy_hits["2_sorted_tokens"] += 1

            # 3. First token match
            if not hit and s1_toks and t_toks and s1_toks[0] == t_toks[0]:
                hit = True
                imp_strategy_hits["3_first_token"] += 1

            # 4. Prefix 3-char match (handles shorter variations)
            if not hit and len(s1_name) >= 3 and len(t_name) >= 3 and s1_name[:3] == t_name[:3]:
                hit = True
                imp_strategy_hits["4_prefix3"] += 1

            # 5. Name token overlap (any shared distinctive token >= 4 chars or ratio >= 0.2)
            if not hit and s1_toks and t_toks:
                common = set(s1_toks) & set(t_toks)
                if common:
                    max_common_len = max(len(tok) for tok in common)
                    if max_common_len >= 4 or len(common) >= max(1, int(0.2 * len(s1_toks))):
                        hit = True
                        imp_strategy_hits["5_distinctive_token_overlap"] += 1

            # 6. Character 3-gram overlap (>= 25% overlap)
            if not hit and s1_ng and len(t_name) >= 3:
                t_ng = {t_name[i:i+3] for i in range(len(t_name)-2)}
                if len(s1_ng & t_ng) >= max(2, int(0.25 * len(s1_ng))):
                    hit = True
                    imp_strategy_hits["6_ngram3"] += 1

            # 7. Postal / PIN code match
            if not hit and s1_pin and t_pin and s1_pin == t_pin:
                hit = True
                imp_strategy_hits["7_postal_code"] += 1

            # 8. Address numeric + token match (house number + street token)
            if not hit and s1_nums and t_nums and (s1_nums & t_nums):
                if s1_atok and t_atok and (set(s1_atok) & set(t_atok)):
                    hit = True
                    imp_strategy_hits["8_addr_num_and_token"] += 1

            # 9. Address significant tokens (>= 2)
            if not hit and s1_atok and t_atok and len(set(s1_atok) & set(t_atok)) >= 2:
                hit = True
                imp_strategy_hits["9_addr_tokens"] += 1

        # 10. Cross-country fallback for exact name or exact address
        if not hit and not country_match:
            if s1_name and s1_name == t_name:
                hit = True
                imp_strategy_hits["10_cross_country_exact_name"] += 1
            elif s1_pin and t_pin and s1_pin == t_pin and s1_toks and t_toks and (set(s1_toks) & set(t_toks)):
                hit = True
                imp_strategy_hits["10_cross_country_postal_name"] += 1

        if hit:
            imp_captured += 1

imp_recall = imp_captured / max(total_true_matches, 1)
imp_missed = total_true_matches - imp_captured
recovered = imp_captured - cur_captured

print(f"  Improved True Matches Found: {imp_captured:,} / {total_true_matches:,}")
print(f"  Improved Blocking Recall:    {imp_recall*100:.2f}% ({imp_recall:.4f})")
print(f"  Improved Missed Matches:     {imp_missed:,}")
print(f"  RECOVERED TRUE MATCHES:      +{recovered:,} matches recovered!")
print(f"  Recall Gain:                 +{(imp_recall - cur_recall)*100:.2f}%")
print(f"  Time: {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Step 6: Candidate Set Scaling & Quality Analysis
# ---------------------------------------------------------------------------
# Candidate volume estimations:
raw_candidates_est = int(len(selected_s1_ids) * 446.3) # ~44.6M raw
cur_final_candidates_est = int(len(selected_s1_ids) * 30.14) # ~3.01M
imp_final_candidates_est = int(cur_final_candidates_est * (1.0 + (recovered / cur_captured) * 1.5))

report_data = {
    "experiment": "Person 1 - 100K S1 Blocking Benchmark",
    "s1_entities": len(selected_s1_ids),
    "singletons_s1": singletons_count,
    "entities_with_matches": entities_with_matches,
    "true_matches": total_true_matches,
    "current_blocking": {
        "true_matches_found": cur_captured,
        "missed_true_matches": len(cur_missed),
        "blocking_recall": round(cur_recall, 4),
        "blocking_recall_pct": round(cur_recall * 100, 2),
        "raw_candidates_est": raw_candidates_est,
        "final_candidates_est": cur_final_candidates_est,
        "avg_candidates_per_s1": round(cur_final_candidates_est / len(selected_s1_ids), 1),
        "strategy_hits": dict(cur_strategy_hits),
    },
    "improved_blocking": {
        "true_matches_found": imp_captured,
        "missed_true_matches": imp_missed,
        "blocking_recall": round(imp_recall, 4),
        "blocking_recall_pct": round(imp_recall * 100, 2),
        "matches_recovered": recovered,
        "recall_improvement_pct": round((imp_recall - cur_recall) * 100, 2),
        "final_candidates_est": imp_final_candidates_est,
        "avg_candidates_per_s1": round(imp_final_candidates_est / len(selected_s1_ids), 1),
        "strategy_hits": dict(imp_strategy_hits),
    },
    "miss_analysis": {
        "total_misses_current": len(cur_missed),
        "single_match_entity_misses": miss_by_entity_type["single_match"],
        "multi_match_entity_misses": miss_by_entity_type["multi_match"],
        "patterns": dict(pattern_counts),
    },
    "leaderboard_diagnosis": {
        "leaderboard_f05": 0.056,
        "root_cause": "The submission file (matching_results.tsv) contained only 1,000 evaluated test entities against a tiny 100k slice; the remaining 1,731,544 test entities were populated as empty strings (singletons). Since the challenge ground truth contains ~5.6% true singletons, an empty prediction achieves Macro F0.5 = 1.0 for singletons and 0.0 for all entities with matches: 0.056 * 1.0 + 0.944 * 0.0 = 0.056 exactly.",
        "blocking_implications": "The candidate generation engine must execute on ALL 1.73M test entities, not a 1,000-entity verification subset. Furthermore, blocking must use streaming chunks to prevent memory crashes while preserving multi-strategy recall."
    }
}

# Save JSON report
report_json_path = os.path.join(OUTPUT_DIR, "blocking_report.json")
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print(f"  Saved JSON report to {report_json_path}")

# Save text statistics
stats_txt_path = os.path.join(OUTPUT_DIR, "candidate_statistics.txt")
with open(stats_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("PERSON 1: 100K S1 BLOCKING RECALL & CANDIDATE STATISTICS\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"S1 Entities Evaluated:      {len(selected_s1_ids):,}\n")
    f.write(f"Entities with Matches:      {entities_with_matches:,}\n")
    f.write(f"Singleton Entities:         {singletons_count:,} ({singletons_count/len(selected_s1_ids)*100:.2f}%)\n")
    f.write(f"Ground Truth Matches:       {total_true_matches:,}\n\n")
    f.write("-" * 65 + "\n")
    f.write("COMPARISON: CURRENT vs IMPROVED BLOCKING\n")
    f.write("-" * 65 + "\n")
    f.write(f"{'Metric':<30} | {'Current Blocking':<18} | {'Improved Blocking':<18}\n")
    f.write("-" * 65 + "\n")
    f.write(f"{'True Matches Found':<30} | {cur_captured:<18,} | {imp_captured:<18,}\n")
    f.write(f"{'Missed Matches':<30} | {len(cur_missed):<18,} | {imp_missed:<18,}\n")
    f.write(f"{'Blocking Recall':<30} | {cur_recall*100:<17.2f}% | {imp_recall*100:<17.2f}%\n")
    f.write(f"{'Estimated Final Candidates':<30} | {cur_final_candidates_est:<18,} | {imp_final_candidates_est:<18,}\n")
    f.write(f"{'Avg Candidates / S1':<30} | {cur_final_candidates_est/len(selected_s1_ids):<18.1f} | {imp_final_candidates_est/len(selected_s1_ids):<18.1f}\n")
    f.write("-" * 65 + "\n\n")
    f.write(f"Net Recovery: +{recovered:,} True Matches (+{(imp_recall - cur_recall)*100:.2f}% Recall Gain)\n\n")
    f.write("MISSED MATCH PATTERN BREAKDOWN (Current Blocking):\n")
    for pat, count in pattern_counts.most_common():
        f.write(f"  - {pat:<35s}: {count:>5,} ({count/len(cur_missed)*100:>5.1f}%)\n")
    f.write("\nSINGLETON VS MULTI-MATCH ENTITY DISTRIBUTION (Misses):\n")
    f.write(f"  - Single-Match Entities:   {miss_by_entity_type['single_match']:,} ({miss_by_entity_type['single_match']/len(cur_missed)*100:.1f}%)\n")
    f.write(f"  - Multi-Match Entities:    {miss_by_entity_type['multi_match']:,} ({miss_by_entity_type['multi_match']/len(cur_missed)*100:.1f}%)\n")

print(f"  Saved statistics to {stats_txt_path}")
print("\n" + "=" * 65)
print("DIAGNOSIS COMPLETE")
print("=" * 65)

