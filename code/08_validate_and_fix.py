"""
08_validate_and_fix.py
=======================
Run the official submission validator and fix any format issues.

Usage:
  python3 code/08_validate_and_fix.py

This script:
1. Checks that output/matching_results.tsv exists
2. Checks that output/candidate_pairs.tsv exists (for test submission, use test_candidate_pairs.tsv)
3. Runs the official validate_submission.py
4. Reports any errors/warnings

For test submission, the candidate_pairs.tsv must be the test version.
This script also copies test_candidate_pairs.tsv → candidate_pairs.tsv if needed.
"""

import os
import sys
import shutil
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(ROOT, "output")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
VALIDATOR = os.path.join(ROOT, "utils", "validate_submission.py")

MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
TEST_CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "test_candidate_pairs.tsv")


def check_file_exists(path: str, label: str) -> bool:
    if not os.path.isfile(path):
        print(f"  ERROR: {label} not found at {path}")
        return False
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  OK: {label} found ({size_mb:.1f} MB)")
    return True


def main():
    print("=" * 60)
    print("Submission Validation")
    print("=" * 60)

    # Check files
    ok = True
    ok &= check_file_exists(MATCHING_RESULTS, "matching_results.tsv")
    ok &= check_file_exists(VALIDATOR, "validate_submission.py")

    # If test_candidate_pairs.tsv exists but candidate_pairs.tsv doesn't, copy it
    if not os.path.isfile(CANDIDATE_PAIRS) and os.path.isfile(TEST_CANDIDATE_PAIRS):
        print(f"\n  Copying test_candidate_pairs.tsv → candidate_pairs.tsv ...")
        shutil.copy2(TEST_CANDIDATE_PAIRS, CANDIDATE_PAIRS)
        print(f"  Done.")

    if not ok:
        print("\n  Some files missing. Please run the full pipeline first.")
        return 1

    # Run official validator
    print(f"\n[Validator] Running official validate_submission.py ...")
    python_exe = sys.executable
    cmd = [
        python_exe,
        VALIDATOR,
        "--matching", MATCHING_RESULTS,
        "--candidate", CANDIDATE_PAIRS,
        "--test-dir", TEST_DIR,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print("  " + "-" * 50)

    result = subprocess.run(cmd, capture_output=False, text=True)

    print("  " + "-" * 50)
    if result.returncode == 0:
        print("\n  PASS — Submission is valid.")
    else:
        print("\n  FAIL — Fix the errors above before submitting.")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
