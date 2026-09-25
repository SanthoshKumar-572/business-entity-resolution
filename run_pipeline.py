"""
run_pipeline.py - Entry point to run the full entity resolution pipeline.

Usage:
    # Run full pipeline from scratch
    python run_pipeline.py

    # Resume from cached intermediate results
    python run_pipeline.py --resume

    # Run smoke test first (recommended)
    python run_pipeline.py --smoke-test

    # Run smoke test with custom size
    python run_pipeline.py --smoke-test --n-s1 500 --n-s2 10000 --n-s3 10000
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test on a small sample before the full pipeline",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached intermediate results",
    )
    parser.add_argument(
        "--n-s1", type=int, default=500,
        help="Number of S1 entities for smoke test (default: 500)",
    )
    parser.add_argument(
        "--n-s2", type=int, default=10000,
        help="Number of S2 records for smoke test (default: 10000)",
    )
    parser.add_argument(
        "--n-s3", type=int, default=10000,
        help="Number of S3 records for smoke test (default: 10000)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("Running smoke test...")
        from business_entity_resolution.smoke_test import run_smoke_test
        metrics = run_smoke_test(
            n_s1=args.n_s1,
            n_s2=args.n_s2,
            n_s3=args.n_s3,
        )
        if metrics is None:
            logger.error("Smoke test failed!")
            sys.exit(1)
        logger.info(f"Smoke test passed with F0.5={metrics.get('f05', 0.0):.4f}")
        logger.info("Smoke test complete. Run without --smoke-test for full pipeline.")
    else:
        logger.info("Starting full pipeline...")
        from business_entity_resolution.src.pipeline import run_pipeline
        run_pipeline(resume=args.resume)


if __name__ == "__main__":
    main()
