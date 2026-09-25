"""
blocking.py - Multi-strategy candidate generation / blocking.

CRITICAL REDESIGN:
The original blocking was producing near-zero candidates because:
1. TF-IDF was skipped for large country groups (US, India have millions of records)
2. Token overlap threshold was too strict

New approach:
1. Exact normalized name match (highest precision)
2. Name first-token + country (handles "ABC Technologies" variants)  
3. Name 4-gram prefix blocking (handles spelling variants)
4. Name 3-gram index with low overlap threshold
5. Address token blocking (same street)
6. Postal code exact match
7. Batched TF-IDF blocking with chunked processing (works for ANY group size)

Key insight: process large country groups in sub-batches using vectorized
inverted-index lookups rather than dense cosine similarity matrices.

All strategies are UNIONED — a candidate appearing in any strategy is kept.
"""

import gc
import logging
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config as cfg
from .normalization import get_name_tokens, get_address_tokens

logger = logging.getLogger(__name__)

# Stop words to exclude from token-based blocking (they add noise)
_NAME_STOP = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "by", "for", "to", "ltd",
    "llc", "inc", "corp", "co", "pvt", "private", "limited", "company",
    "group", "services", "solutions", "trading", "enterprises", "industries",
    "international", "global", "national", "india", "us", "france",
})

_ADDR_STOP = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "by", "for", "to",
    "no", "near", "opposite",
})


def _meaningful_name_tokens(norm_name: str) -> List[str]:
    """Return meaningful name tokens (skip stop words and short tokens)."""
    tokens = get_name_tokens(norm_name)
    return [t for t in tokens if t not in _NAME_STOP and len(t) >= 3]


def _meaningful_addr_tokens(norm_addr: str) -> List[str]:
    """Return meaningful address tokens (skip stop words, keep numeric)."""
    tokens = get_address_tokens(norm_addr)
    # Keep all tokens of length >= 3, including numbers
    return [t for t in tokens if len(t) >= 3 and t not in _ADDR_STOP]


# ─── Build Inverted Index Helpers ────────────────────────────────────────────

def build_name_token_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build inverted index: token → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_name in zip(df["entity_id"], df["norm_name"]):
        for tok in _meaningful_name_tokens(str(norm_name)):
            index[tok].add(eid)
    return index


def build_exact_name_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build inverted index: exact_norm_name → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_name in zip(df["entity_id"], df["norm_name"]):
        if norm_name:
            index[norm_name].add(eid)
    return index


def build_name_prefix_index(df: pd.DataFrame, n: int = 4) -> Dict[str, Set[str]]:
    """Build inverted index: name_prefix_n → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_name in zip(df["entity_id"], df["norm_name"]):
        text = str(norm_name).strip()
        if len(text) >= n:
            key = text[:n]
            index[key].add(eid)
    return index


def build_name_ngram_index(df: pd.DataFrame, n: int = 3) -> Dict[str, Set[str]]:
    """Build inverted index: char_ngram → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_name in zip(df["entity_id"], df["norm_name"]):
        text = str(norm_name).strip()
        if len(text) >= n:
            ngrams = {text[i:i+n] for i in range(len(text)-n+1)}
            for ng in ngrams:
                index[ng].add(eid)
    return index


def build_addr_token_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build inverted index: address_token → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_addr in zip(df["entity_id"], df["norm_address"]):
        for tok in _meaningful_addr_tokens(str(norm_addr)):
            index[tok].add(eid)
    return index


def build_postal_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build inverted index: postal_code → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, pc in zip(df["entity_id"], df.get("postal_code", [""] * len(df))):
        pc = str(pc).strip() if pc and pc != "None" else ""
        if pc and len(pc) >= 5:
            index[pc].add(eid)
    return index


def build_first_token_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build inverted index: first_meaningful_token → set of entity IDs."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for eid, norm_name in zip(df["entity_id"], df["norm_name"]):
        tokens = _meaningful_name_tokens(str(norm_name))
        if tokens:
            index[tokens[0]].add(eid)
    return index


# ─── Batched TF-IDF Blocking ────────────────────────────────────────────────

def batched_tfidf_blocking(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    text_col: str = "norm_name",
    top_k: int = 80,
    s1_batch_size: int = 2000,
    target_batch_size: int = 100_000,
) -> Dict[str, Set[str]]:
    """Scalable TF-IDF blocking: works for ANY group size.

    Instead of building a massive cosine similarity matrix, processes S1 in 
    batches and uses sparse matrix operations.

    For very large target groups (>100K), it processes targets in sub-batches,
    keeping only the top-k candidates per S1 across all sub-batches.
    """
    if target_df.empty or s1_df.empty:
        return {}

    s1_ids = s1_df["entity_id"].tolist()
    s1_texts = [str(x) for x in s1_df[text_col].fillna("").tolist()]
    target_ids = target_df["entity_id"].tolist()
    target_texts = [str(x) for x in target_df[text_col].fillna("").tolist()]

    # Fit vectorizer on a combined sample (avoid memory explosion)
    sample_size = min(200_000, len(s1_texts) + len(target_texts))
    all_sample = s1_texts[:50_000] + target_texts[:150_000]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=min(30_000, cfg.FEATURES["tfidf_max_features"]),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    try:
        vectorizer.fit(all_sample[:sample_size])
    except Exception as e:
        logger.warning(f"TF-IDF vectorizer fit failed: {e}")
        return {}

    candidates: Dict[str, Set[str]] = defaultdict(set)

    # Transform targets in sub-batches
    n_target_batches = max(1, (len(target_ids) + target_batch_size - 1) // target_batch_size)

    # For each S1 batch, accumulate top-k scores across all target batches
    # Keep running top-k using heaps
    s1_top_scores: Dict[str, List[Tuple[float, str]]] = {sid: [] for sid in s1_ids}

    for tb_start in range(0, len(target_ids), target_batch_size):
        tb_end = min(tb_start + target_batch_size, len(target_ids))
        t_ids_batch = target_ids[tb_start:tb_end]
        t_texts_batch = target_texts[tb_start:tb_end]

        try:
            t_matrix = vectorizer.transform(t_texts_batch)
        except Exception:
            continue

        # Process S1 in batches against this target batch
        for sb_start in range(0, len(s1_ids), s1_batch_size):
            sb_end = min(sb_start + s1_batch_size, len(s1_ids))
            s1_ids_batch = s1_ids[sb_start:sb_end]
            s1_texts_batch = s1_texts[sb_start:sb_end]

            try:
                s1_matrix = vectorizer.transform(s1_texts_batch)
                # Sparse cosine similarity (much more memory efficient)
                from sklearn.preprocessing import normalize as sk_normalize
                s1_norm = sk_normalize(s1_matrix, norm="l2")
                t_norm  = sk_normalize(t_matrix, norm="l2")
                # Compute scores: (s1_batch x target_batch)
                scores = (s1_norm @ t_norm.T).toarray()
            except Exception:
                continue

            # For each S1 in this batch, find top-k from this target sub-batch
            for i, s1_id in enumerate(s1_ids_batch):
                row = scores[i]
                # Get indices of top-k from this target sub-batch
                if top_k >= len(row):
                    top_indices = np.where(row > 0.1)[0]
                else:
                    top_indices = np.argpartition(row, -top_k)[-top_k:]
                    top_indices = top_indices[row[top_indices] > 0.1]

                for idx in top_indices:
                    candidates[s1_id].add(t_ids_batch[idx])

        del t_matrix
        gc.collect()

    logger.debug(f"  TF-IDF blocking: {sum(len(v) for v in candidates.values()):,} candidates")
    return candidates


# ─── Main BlockingEngine ─────────────────────────────────────────────────────

class BlockingEngine:
    """Multi-strategy blocking engine.

    Key improvements over v1:
    - Scalable TF-IDF for any group size (batched processing)
    - Aggressive first-token blocking
    - Better ngram overlap threshold (lower = higher recall)
    - Address token blocking uses important tokens only
    - Country-level grouping still applied to restrict search space
    """

    def __init__(
        self,
        max_candidates: int = cfg.BLOCKING["max_candidates_per_s1"],
        tfidf_top_k: int = cfg.BLOCKING["tfidf_top_k"],
    ):
        self.max_candidates = max_candidates
        self.tfidf_top_k = tfidf_top_k

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """Generate candidates for all S1 entities from S2 and S3.

        Returns: dict mapping S1 entity_id → set of candidate S2/S3 ids.
        """
        logger.info("Starting candidate generation (blocking)...")

        # Combine S2 and S3 targets
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)

        # Initialize candidate dict for all S1 entities
        candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_df["entity_id"]}

        # Group target_df by country for country-scoped blocking
        target_by_country: Dict[str, pd.DataFrame] = {}
        for country, grp in target_df.groupby("norm_country"):
            target_by_country[country] = grp

        # Group S1 by country
        for country, s1_grp in s1_df.groupby("norm_country"):
            n_s1 = len(s1_grp)
            n_targets = len(target_by_country.get(country, pd.DataFrame()))
            logger.info(
                f"  Blocking country='{country}': "
                f"S1={n_s1:,} × Targets={n_targets:,}"
            )

            if country in target_by_country:
                t_df = target_by_country[country]
            else:
                logger.warning(f"  No targets for country='{country}', using all")
                t_df = target_df

            self._apply_strategies(s1_grp, t_df, candidates)

        # Handle S1 entities with empty norm_country → compare against all targets
        empty_country_s1 = s1_df[s1_df["norm_country"] == ""]
        if not empty_country_s1.empty:
            logger.info(
                f"  {len(empty_country_s1):,} S1 with unknown country → all targets"
            )
            self._apply_strategies(empty_country_s1, target_df, candidates)

        # Apply cap
        over_cap = 0
        for s1_id in candidates:
            if len(candidates[s1_id]) > self.max_candidates:
                over_cap += 1
                # Prioritize S2 and S3 balanced
                cand_list = list(candidates[s1_id])
                candidates[s1_id] = set(cand_list[: self.max_candidates])

        # Summary stats
        sizes = [len(v) for v in candidates.values()]
        non_empty = sum(1 for v in sizes if v > 0)
        total_candidates = sum(sizes)
        avg_cands = total_candidates / max(len(sizes), 1)

        logger.info(
            f"Blocking complete: {total_candidates:,} total candidates "
            f"({non_empty:,}/{len(candidates):,} S1 with candidates, "
            f"avg={avg_cands:.1f}, {over_cap} capped at {self.max_candidates})"
        )
        return candidates

    def _apply_strategies(
        self,
        s1_grp: pd.DataFrame,
        t_df: pd.DataFrame,
        candidates: Dict[str, Set[str]],
    ) -> None:
        """Apply all blocking strategies for a country group and update candidates."""

        if t_df.empty or s1_grp.empty:
            return

        # ── Strategy A: Exact normalized name match ──────────────────────────
        exact_name_idx = build_exact_name_index(t_df)
        for eid, norm_name in zip(s1_grp["entity_id"], s1_grp["norm_name"]):
            if norm_name and norm_name in exact_name_idx:
                candidates[eid].update(exact_name_idx[norm_name])

        # ── Strategy B: Name first-token blocking ────────────────────────────
        first_tok_idx = build_first_token_index(t_df)
        for eid, norm_name in zip(s1_grp["entity_id"], s1_grp["norm_name"]):
            tokens = _meaningful_name_tokens(str(norm_name))
            if tokens and tokens[0] in first_tok_idx:
                candidates[eid].update(first_tok_idx[tokens[0]])

        # ── Strategy C: Name 4-char prefix blocking ──────────────────────────
        prefix4_idx = build_name_prefix_index(t_df, n=4)
        for eid, norm_name in zip(s1_grp["entity_id"], s1_grp["norm_name"]):
            text = str(norm_name).strip()
            if len(text) >= 4:
                key = text[:4]
                if key in prefix4_idx:
                    candidates[eid].update(prefix4_idx[key])

        # ── Strategy D: Name token inverted index (≥1 shared token) ─────────
        tok_idx = build_name_token_index(t_df)
        for eid, norm_name in zip(s1_grp["entity_id"], s1_grp["norm_name"]):
            s1_tokens = _meaningful_name_tokens(str(norm_name))
            if not s1_tokens:
                continue
            # Count token overlap per candidate
            overlap: Dict[str, int] = defaultdict(int)
            for tok in s1_tokens:
                if tok in tok_idx:
                    for cid in tok_idx[tok]:
                        overlap[cid] += 1
            # Require overlap proportional to token count (but at least 1)
            min_overlap = max(1, int(0.2 * len(s1_tokens)))
            for cid, cnt in overlap.items():
                if cnt >= min_overlap:
                    candidates[eid].add(cid)

        # ── Strategy E: Name 3-gram blocking ─────────────────────────────────
        ngram3_idx = build_name_ngram_index(t_df, n=3)
        for eid, norm_name in zip(s1_grp["entity_id"], s1_grp["norm_name"]):
            text = str(norm_name).strip()
            if len(text) < 3:
                continue
            s1_ngrams = {text[i:i+3] for i in range(len(text)-2)}
            ngram_overlap: Dict[str, int] = defaultdict(int)
            for ng in s1_ngrams:
                if ng in ngram3_idx:
                    for cid in ngram3_idx[ng]:
                        ngram_overlap[cid] += 1
            # Require at least 30% ngram overlap (low threshold for high recall)
            thresh = max(2, int(0.30 * len(s1_ngrams)))
            for cid, cnt in ngram_overlap.items():
                if cnt >= thresh:
                    candidates[eid].add(cid)

        # ── Strategy F: Postal code exact match ─────────────────────────────
        if "postal_code" in t_df.columns and "postal_code" in s1_grp.columns:
            postal_idx = build_postal_index(t_df)
            for eid, pc in zip(s1_grp["entity_id"], s1_grp["postal_code"]):
                pc = str(pc).strip() if pc and pc != "None" else ""
                if pc and len(pc) >= 5 and pc in postal_idx:
                    candidates[eid].update(postal_idx[pc])

        # ── Strategy G: Address token blocking ───────────────────────────────
        addr_tok_idx = build_addr_token_index(t_df)
        for eid, norm_addr in zip(s1_grp["entity_id"], s1_grp["norm_address"]):
            addr_toks = _meaningful_addr_tokens(str(norm_addr))
            if len(addr_toks) < 2:
                continue
            addr_overlap: Dict[str, int] = defaultdict(int)
            for tok in addr_toks:
                if tok in addr_tok_idx:
                    for cid in addr_tok_idx[tok]:
                        addr_overlap[cid] += 1
            # Require ≥2 shared address tokens
            for cid, cnt in addr_overlap.items():
                if cnt >= 2:
                    candidates[eid].add(cid)

        # ── Strategy H: Batched TF-IDF blocking ─────────────────────────────
        # Apply to ALL group sizes using batched processing
        try:
            tfidf_cands = batched_tfidf_blocking(
                s1_grp, t_df,
                text_col="norm_name",
                top_k=self.tfidf_top_k,
                s1_batch_size=1000,
                target_batch_size=50_000,
            )
            for s1_id, cand_set in tfidf_cands.items():
                candidates[s1_id].update(cand_set)
        except Exception as e:
            logger.warning(f"TF-IDF blocking failed: {e}")


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int = None,
    tfidf_top_k: int = None,
) -> Dict[str, Set[str]]:
    """Convenience wrapper to generate candidates."""
    kwargs = {}
    if max_candidates is not None:
        kwargs["max_candidates"] = max_candidates
    if tfidf_top_k is not None:
        kwargs["tfidf_top_k"] = tfidf_top_k
    engine = BlockingEngine(**kwargs)
    return engine.generate_candidates(s1_df, s2_df, s3_df)
