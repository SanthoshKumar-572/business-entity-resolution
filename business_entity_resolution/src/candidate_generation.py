"""
candidate_generation.py - Candidate generation and multi-strategy blocking.
"""

from .blocking import (
    BlockingEngine,
    generate_candidates,
    batched_tfidf_blocking,
    build_exact_name_index,
    build_name_prefix_index,
    build_name_ngram_index,
    build_name_token_index,
    build_addr_token_index,
    build_postal_index,
    build_first_token_index,
)

__all__ = [
    "BlockingEngine",
    "generate_candidates",
    "batched_tfidf_blocking",
    "build_exact_name_index",
    "build_name_prefix_index",
    "build_name_ngram_index",
    "build_name_token_index",
    "build_addr_token_index",
    "build_postal_index",
    "build_first_token_index",
]
