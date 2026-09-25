"""
inference.py - Full test inference pipeline for Business Entity Resolution.
"""

from .predict import predict_test, run_validator, _compute_proba_in_chunks

__all__ = [
    "predict_test",
    "run_validator",
    "_compute_proba_in_chunks",
]
