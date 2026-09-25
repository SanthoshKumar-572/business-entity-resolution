"""
run_pipeline.py
================
Master pipeline runner. Executes all steps in order.

Usage:
  python3 code/run_pipeline.py [--mode train|test|full]

Modes:
  train  — Run training pipeline only (steps 0–4)
  test   — Run test inference pipeline only (steps 5–8)
  full   — Run complete pipeline end-to-end (default)

Steps:
  0. (Optional) Split S1 if S1_Laptop2.tsv not present
  1. Generate candidates for Laptop 2 (train S1)
  2. Merge candidates (laptop1 + laptop2)
  3. Feature engineering (train)
  4. Train LightGBM model + threshold tuning
  5. Generate test candidates
  6. Feature engineering (test)
  7. Final inference
  8. Validate submission

Each step is skipped with a warning if its expected input files are missing.
"""

import os
import sys
import subprocess
import argparse
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

PYTHON = sys.executable


def run_step(step_num: int, script: str, args: list = None, description: str = ""):
    """Run a pipeline step. Returns True on success."""
    script_path = os.path.join(SCRIPT_DIR, script)
    cmd = [PYTHON, script_path] + (args or [])

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    start = time.time()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ Step {step_num} complete ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  ✗ Step {step_num} FAILED (returncode={result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline Runner")
    parser.add_argument("--mode", choices=["train", "test", "full"], default="full",
                        help="Pipeline mode: train, test, or full (default: full)")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip S1 split step (if S1_Laptop2.tsv already exists)")
    args = parser.parse_args()

    print("=" * 60)
    print("Business Entity Resolution Pipeline")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    all_ok = True
    pipeline_start = time.time()

    if args.mode in ("train", "full"):
        # Step 0: Split S1 (only if needed)
        if not args.skip_split:
            s1_laptop2 = os.path.join(ROOT, "dataset", "train", "S1_Laptop2.tsv")
            if not os.path.isfile(s1_laptop2):
                ok = run_step(0, "07_split_s1_laptop2.py", description="Split S1 for Laptop 2")
                all_ok &= ok
            else:
                print(f"\nStep 0: Skipped (S1_Laptop2.tsv already exists)")

        # Step 1: Generate candidates for laptop2
        ok = run_step(1, "01_generate_candidates_laptop2.py",
                      description="Generate candidates for S1_Laptop2")
        all_ok &= ok

        # Step 2: Merge candidates
        ok = run_step(2, "02_merge_candidates.py",
                      description="Merge laptop1 + laptop2 candidates")
        all_ok &= ok

        # Step 3: Feature engineering (train)
        ok = run_step(3, "03_feature_engineering.py", args=["train"],
                      description="Feature engineering (training data)")
        all_ok &= ok

        # Step 4: Train model
        ok = run_step(4, "04_train_model.py",
                      description="Train LightGBM + threshold tuning")
        all_ok &= ok

    if args.mode in ("test", "full"):
        # Step 5: Generate test candidates
        ok = run_step(5, "05_generate_test_candidates.py",
                      description="Generate test candidates")
        all_ok &= ok

        # Step 6: Feature engineering (test)
        ok = run_step(6, "03_feature_engineering.py", args=["test"],
                      description="Feature engineering (test data)")
        all_ok &= ok

        # Step 7: Final inference
        ok = run_step(7, "06_final_inference.py",
                      description="Final inference → matching_results.tsv")
        all_ok &= ok

        # Step 8: Validate
        ok = run_step(8, "08_validate_and_fix.py",
                      description="Validate submission")
        # Don't fail the pipeline if validation fails — just report

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"Pipeline {'COMPLETE' if all_ok else 'FINISHED WITH ERRORS'}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
