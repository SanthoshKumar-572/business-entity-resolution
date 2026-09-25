# Business Entity Resolution — Solution README

## Team Setup

This is a **2-person team** solution.

| Person | Responsibility | Data |
|--------|---------------|------|
| Person 1 (Laptop 1) | S1 first half candidate generation | `S1_Laptop1.tsv` |
| Person 2 (Laptop 2) | Feature engineering, ML training, full inference | `S1_Laptop2.tsv` + all S2/S3 |

---

## Directory Structure

```
student_resource/
├── code/                           # All pipeline scripts (commit this)
│   ├── normalize.py                # Shared normalization utilities
│   ├── 01_generate_candidates_laptop2.py
│   ├── 02_merge_candidates.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py
│   ├── 05_generate_test_candidates.py
│   ├── 06_final_inference.py
│   ├── 07_split_s1_laptop2.py
│   ├── 08_validate_and_fix.py
│   └── run_pipeline.py             # Master runner
├── dataset/
│   ├── train/                      # Training data (NOT in git)
│   │   ├── S1_Laptop2.tsv
│   │   ├── train_source2.tsv
│   │   └── train_source3.tsv
│   └── test/                       # Test data (NOT in git)
├── output/                         # Generated outputs (NOT in git)
│   ├── candidates_laptop2.tsv
│   ├── candidates_laptop1.tsv      # Received from Person 1 via Drive/USB
│   ├── candidate_pairs.tsv         # Merged (SUBMIT THIS)
│   ├── test_candidate_pairs.tsv
│   ├── features_train.parquet
│   ├── features_test.parquet
│   └── matching_results.tsv        # FINAL SUBMISSION FILE
├── model/                          # Trained model (NOT in git — too large)
│   ├── lgbm_model.txt
│   ├── threshold.txt
│   ├── feature_importance.csv
│   └── validation_metrics.json
├── utils/
│   └── validate_submission.py      # Official validator (provided)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Pipeline Overview

### Step 0 — S1 Split (if needed)

If `dataset/train/S1_Laptop2.tsv` doesn't exist:

```bash
python3 code/07_split_s1_laptop2.py
```

This takes the sorted second half (50%) of `train_source1.tsv`.

### Step 1 — Candidate Generation (Laptop 2's S1 portion)

```bash
python3 code/01_generate_candidates_laptop2.py
```

Outputs: `output/candidates_laptop2.tsv`  
Columns: `source1_entity_id | candidate_entity_id`

**Blocking strategy:** 6 blocking key types per entity:
1. `name_prefix4_country` — first 4 chars of name first token + country
2. `name_prefix4_2_country` — first 4 chars of second token + country
3. `name_bigram_country` — sorted 6-char prefixes of first two name tokens
4. `addr_num_name_prefix` — first address number + name prefix + country
5. `addr_prefix_country` — first 2 significant address tokens
6. `name_token0_exact_country` — exact first token + country

Each blocking key maps to at most 200 candidates (hard cap).

### Step 2 — Merge Candidates

**After receiving `candidates_laptop1.tsv` from Person 1** (via Drive/USB/network):

```bash
# Place candidates_laptop1.tsv in output/
python3 code/02_merge_candidates.py
```

Outputs: `output/candidate_pairs.tsv`  
Columns: `source1_entity_id | candidate_entity_ids` (comma-separated)

- Deduplicates pairs
- Verifies all IDs are S2- or S3- prefixed
- Adds empty rows for S1 entities with no candidates

### Step 3 — Feature Engineering (Training)

```bash
python3 code/03_feature_engineering.py train
```

Outputs: `output/features_train.parquet`

**Features computed per candidate pair:**

| Category | Features |
|----------|----------|
| Name | exact_match, token_overlap, jaccard_3gram, char_sim, prefix_sim, len_diff, fuzzy_ratio, fuzzy_partial, fuzzy_token_sort, fuzzy_token_set |
| Address | exact_match, token_overlap, jaccard_3gram, char_sim, numeric_overlap, prefix_sim, len_diff, fuzzy_ratio |
| Country | exact_match, norm_match |
| Combined | name_addr_combined, missing indicators (4), source indicator (2) |

Total: **26 features**

### Step 4 — Model Training + Threshold Tuning

```bash
python3 code/04_train_model.py
```

- Splits by S1 entity (80% train / 20% val — no leakage)
- Trains LightGBM with `scale_pos_weight` for class imbalance
- Evaluates F0.5 at thresholds: 0.50 → 0.98
- Saves model + best threshold to `model/`

### Step 5 — Test Candidate Generation

```bash
python3 code/05_generate_test_candidates.py
```

Outputs: `output/test_candidate_pairs.tsv`  
Same blocking logic as training.

### Step 6 — Feature Engineering (Test)

```bash
python3 code/03_feature_engineering.py test
```

Outputs: `output/features_test.parquet`

### Step 7 — Final Inference

```bash
python3 code/06_final_inference.py
```

Outputs: `output/matching_results.tsv`  
Applies threshold, generates one row per test S1 entity.

### Step 8 — Validate

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Or use the wrapper:

```bash
python3 code/08_validate_and_fix.py
```

---

## Full Pipeline (One Command)

```bash
# Full pipeline
python3 code/run_pipeline.py --mode full

# Training only
python3 code/run_pipeline.py --mode train

# Test inference only (after training)
python3 code/run_pipeline.py --mode test
```

---

## Normalization

Business names:
- Unicode NFKC normalization
- Lowercase
- Expand `&` → `and`
- Remove punctuation (keep alphanumeric + Devanagari + Latin extended)
- Abbreviate legal suffixes: `incorporated` → `inc`, `limited` → `ltd`, etc.
- Collapse whitespace

Addresses:
- Unicode NFKC normalization
- Lowercase
- Remove punctuation
- Abbreviate: `street` → `st`, `road` → `rd`, `avenue` → `ave`, etc.
- Directional abbreviations: `north` → `n`, `south` → `s`

Country:
- Lowercase + strip

---

## Model

**Algorithm:** LightGBM (gradient boosted trees)  
**Objective:** Binary classification  
**Loss:** Binary cross-entropy  
**Class imbalance:** `scale_pos_weight = neg_count / pos_count`  
**Early stopping:** 50 rounds on validation logloss  
**Hyperparameters:** `num_leaves=127`, `lr=0.05`, `subsample=0.8`, `colsample=0.8`

---

## Evaluation

**Metric:** F0.5 (precision-weighted) at entity level  

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed per S1 entity, then macro-averaged
- Singletons (no true matches): score 1.0 if predicted empty, 0.0 otherwise
- Split: 80% train / 20% val by S1 entity (not by candidate pair)

---

## Person 1 Coordination

Person 1 generates: `output/candidates_laptop1.tsv`

Transfer method: Google Drive / OneDrive / USB

**NEVER commit candidate files to git.** They are too large and contain derived data.

---

## Important Notes

- Never commit `dataset/`, `output/`, `model/`, or `*.tsv` files to git
- The `.gitignore` excludes these automatically
- Test set includes `France` country — do not hardcode country filtering
- Every test S1 entity must appear in `matching_results.tsv` (even with empty match)
- F0.5 rewards precision over recall — prefer higher thresholds

---

## How to Reproduce Results

1. Install dependencies: `pip install -r requirements.txt`
2. Place data in `dataset/train/` and `dataset/test/`
3. Place `candidates_laptop1.tsv` in `output/` (from Person 1)
4. Run: `python3 code/run_pipeline.py --mode full`
5. Submit: `output/matching_results.tsv` to leaderboard
