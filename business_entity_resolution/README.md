# Business Entity Resolution Pipeline

## Overview

Complete, scalable, memory-efficient entity resolution pipeline for the Amazon Business Entity Resolution Challenge.

**Goal**: For every Source 1 business entity, identify all matching entities from Source 2 and Source 3.

**Target Metric**: Macro-averaged F0.5 > 0.91

## Architecture

```
RAW DATA (12M+ records)
   ↓
TEXT NORMALIZATION (name, address, country)
   ↓
8-STRATEGY BLOCKING (country-scoped, inverted indexes + TF-IDF)
   ↓
CANDIDATE PAIRS (~300-500 per S1 entity)
   ↓
32 SIMILARITY FEATURES (name, address, postal, country)
   ↓
LIGHTGBM CLASSIFIER (F0.5 optimized)
   ↓
THRESHOLD TUNING (F0.5 maximization on validation)
   ↓
matching_results.tsv + candidate_pairs.tsv
```

## Project Structure

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv    (2,206,822 rows)
│   │   ├── train_source2.tsv    (5,034,617 rows)
│   │   ├── train_source3.tsv    (5,285,604 rows)
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── business_entity_resolution/
│   ├── src/
│   │   ├── config.py            # All configuration
│   │   ├── data_loader.py       # TSV loading
│   │   ├── normalization.py     # Text normalization
│   │   ├── blocking.py          # 8-strategy blocking
│   │   ├── features.py          # 32 similarity features
│   │   ├── train.py             # LightGBM/XGBoost training
│   │   ├── evaluate.py          # F0.5 metric
│   │   ├── threshold.py         # Threshold optimization
│   │   ├── pipeline.py          # Main orchestrator
│   │   ├── predict.py           # Test inference
│   │   ├── output_writer.py     # TSV output
│   │   └── labeling.py          # Train/val split
│   ├── models/                  # Saved models
│   ├── cache/                   # Intermediate results
│   ├── output/                  # Output files
│   └── smoke_test.py
├── output/                      # Final submission files
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py
├── run_pipeline.py              # Main entry point
└── diagnose_blocking.py         # Blocking recall diagnostic
```

## Quick Start

### Step 1: Smoke test (5 minutes)
```powershell
cd student_resource
python run_pipeline.py --smoke-test --n-s1 500 --n-s2 10000 --n-s3 10000
```

### Step 2: Blocking recall diagnosis (30-60 min)
```powershell
python diagnose_blocking.py --n-s1 1000
```

### Step 3: Full pipeline (~3-6 hours)
```powershell
python run_pipeline.py
```

### Step 4: Resume if interrupted
```powershell
python run_pipeline.py --resume
```

### Step 5: Validate submission
```powershell
python utils/validate_submission.py ^
  --matching output/matching_results.tsv ^
  --candidate output/candidate_pairs.tsv ^
  --test-dir dataset/test
```

## Blocking Strategies

The blocking engine uses 8 strategies to maximize recall:

| Strategy | Description |
|----------|-------------|
| A | Exact normalized name match |
| B | First meaningful name token |
| C | Name 4-char prefix |
| D | Name token overlap (≥20% shared) |
| E | Name 3-gram overlap (≥30% shared) |
| F | Postal code exact match |
| G | Address token overlap (≥2 shared) |
| H | Batched TF-IDF cosine (top-80) |

All results are **unioned** to maximize recall.

## Feature Engineering (32 features)

### Name Features
- `name_exact_match` - Exact normalized match
- `name_levenshtein` - Normalized Levenshtein similarity
- `name_jaro_winkler` - Jaro-Winkler similarity
- `name_token_sort` - Token sort ratio
- `name_token_set` - Token set ratio
- `name_jaccard` - Token Jaccard similarity
- `name_char2gram` / `name_char3gram` / `name_char4gram` - Character n-gram similarity
- `name_prefix5` / `name_prefix3` - Prefix similarity
- `name_tfidf_cosine` - TF-IDF cosine similarity
- `name_common_tokens` - Shared token count
- `name_len_diff` - Length difference ratio
- `name_token_count_diff` - Token count difference
- `name_raw_levenshtein` - Raw (pre-normalization) Levenshtein

### Address Features
- `addr_exact_match`, `addr_levenshtein`, `addr_jaccard`
- `addr_char3gram`, `addr_token_sort`, `addr_tfidf_cosine`
- `addr_common_tokens`, `addr_len_diff`
- `addr_numeric_overlap`, `addr_numeric_exact` - House number matching
- `postal_exact_match`, `postal_prefix_match` - PIN/ZIP matching

### Country Features
- `country_exact_match`, `country_sim`

### Cross Features
- `name_and_addr_both_high` - Both name AND address similar
- `name_max_sim` - Max across all name similarity metrics

## Model

- **Primary**: LightGBM (n_estimators=1000, num_leaves=127)
- **Fallback**: XGBoost
- **Loss**: Binary cross-entropy with scale_pos_weight
- **Threshold**: Optimized for macro F0.5 on validation set

## Memory Design

- Country-based grouping limits comparison space
- Batched TF-IDF (2K S1 × 50K target batches)
- Feature computation in 100K pair chunks
- Intermediate results cached to disk
- Pipeline is fully restartable

## Competition Rules

✅ Only uses provided competition data  
✅ No external APIs or databases  
✅ No geocoding or business registries  
✅ Output format validated before submission  
