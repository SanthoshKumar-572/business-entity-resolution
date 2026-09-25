"""
07_split_s1_laptop2.py
=======================
Helper script to split train_source1.tsv into S1_Laptop2.tsv.

Since the competition provides the full train_source1.tsv but my laptop
only processes the second half (Person 2), this script creates S1_Laptop2.tsv
if it doesn't exist.

Usage:
  python3 code/07_split_s1_laptop2.py [--fraction 0.5] [--seed 42]

The split is deterministic by S1 entity ID to ensure reproducibility.
Person 1 and Person 2 must use the SAME seed and complementary fractions.

By default: Person 2 gets the SECOND 50% of S1 entities (sorted by ID).
"""

import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
OUTPUT = os.path.join(TRAIN_DIR, "S1_Laptop2.tsv")


def main():
    parser = argparse.ArgumentParser(description="Split S1 for Person 2's laptop")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="Fraction for Person 1 (default: 0.5 → Person 2 gets the other half)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if os.path.isfile(OUTPUT):
        print(f"S1_Laptop2.tsv already exists at {OUTPUT}. Skipping split.")
        return 0

    print(f"Loading entity IDs from {SOURCE1} ...")
    ids = []
    reader = pd.read_csv(SOURCE1, sep="\t", dtype=str, chunksize=500_000, usecols=["entity_id"])
    for chunk in tqdm(reader, desc="  Loading"):
        ids.extend(chunk["entity_id"].dropna().tolist())
        del chunk

    # Sort for reproducibility
    ids_sorted = sorted(ids)
    split_idx = int(len(ids_sorted) * args.fraction)
    laptop2_ids = set(ids_sorted[split_idx:])  # second half

    print(f"Total S1 entities: {len(ids_sorted):,}")
    print(f"Person 2 (Laptop 2) entities: {len(laptop2_ids):,} ({(1-args.fraction)*100:.0f}%)")

    # Write filtered rows
    print(f"Writing {OUTPUT} ...")
    header_written = False
    total_written = 0

    reader = pd.read_csv(SOURCE1, sep="\t", dtype=str, chunksize=500_000)
    with open(OUTPUT, "w", encoding="utf-8") as f_out:
        for chunk in tqdm(reader, desc="  Filtering"):
            filtered = chunk[chunk["entity_id"].isin(laptop2_ids)]
            if not filtered.empty:
                filtered.to_csv(
                    f_out,
                    sep="\t",
                    index=False,
                    header=not header_written,
                )
                header_written = True
                total_written += len(filtered)
            del chunk

    print(f"Done. Written {total_written:,} rows → {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
