"""
features.py - Feature engineering for candidate pairs.

For each (S1, S2/S3) candidate pair, generates rich similarity features:
  - Business name similarity (multiple metrics)
  - Address similarity (multiple metrics)
  - Country match
  - Structural features

All computations are local — no external APIs.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# rapidfuzz for fast string similarity
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler as _JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz not available; using basic string similarity")

from . import config as cfg
from .normalization import get_name_tokens, get_address_tokens, extract_numeric_tokens

logger = logging.getLogger(__name__)


# ─── String Similarity Utilities ─────────────────────────────────────────────

def levenshtein_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a, b) / 100.0
    # Fallback: simple ratio
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    # Simple character-level ratio
    matches = sum(c1 == c2 for c1, c2 in zip(a, b))
    return 2.0 * matches / (len(a) + len(b))


def jaro_winkler_sim(a: str, b: str) -> float:
    """Jaro-Winkler similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return _JaroWinkler.similarity(a, b)
    return levenshtein_sim(a, b)  # fallback


def token_sort_ratio(a: str, b: str) -> float:
    """Token sort ratio (order-insensitive similarity)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.token_sort_ratio(a, b) / 100.0
    # Fallback: sort tokens and compare
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return levenshtein_sim(a_sorted, b_sorted)


def token_set_ratio(a: str, b: str) -> float:
    """Token set ratio (subset-insensitive similarity)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.token_set_ratio(a, b) / 100.0
    return token_sort_ratio(a, b)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def char_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a_ngrams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_ngrams = {b[i:i+n] for i in range(len(b)-n+1)}
    return jaccard_similarity(a_ngrams, b_ngrams)


def common_token_count(a: str, b: str) -> int:
    """Count of shared tokens between two strings."""
    a_toks = set(a.split())
    b_toks = set(b.split())
    return len(a_toks & b_toks)


def prefix_similarity(a: str, b: str, n: int = 5) -> float:
    """Similarity of first n characters."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    pa = a[:n]
    pb = b[:n]
    return levenshtein_sim(pa, pb)


# ─── TF-IDF Cosine Feature ────────────────────────────────────────────────────

class TFIDFFeatureComputer:
    """Compute TF-IDF cosine similarity for pairs of text strings.

    Fitted once on the combined corpus of training/test records.
    """

    def __init__(self, analyzer="char_wb", ngram_range=(2, 4), max_features=50000):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=1,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: List[str]):
        """Fit on a corpus of texts."""
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def cosine_sim(self, a: str, b: str) -> float:
        """Compute cosine similarity between two texts."""
        if not self._fitted:
            raise RuntimeError("TFIDFFeatureComputer not fitted yet")
        vecs = self.vectorizer.transform([a or "", b or ""])
        score = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return float(score)

    def batch_cosine_sim(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Compute cosine similarity for a batch of (a, b) pairs efficiently.
        Uses matrix multiply for true vectorization.
        """
        if not self._fitted:
            raise RuntimeError("TFIDFFeatureComputer not fitted yet")
        if not pairs:
            return np.array([], dtype=np.float32)
        a_texts = [p[0] or "" for p in pairs]
        b_texts = [p[1] or "" for p in pairs]
        a_mat = self.vectorizer.transform(a_texts)
        b_mat = self.vectorizer.transform(b_texts)
        # Row-wise dot product then normalize by row norms
        from sklearn.preprocessing import normalize
        a_norm = normalize(a_mat, norm="l2")
        b_norm = normalize(b_mat, norm="l2")
        # Element-wise row dot products
        scores = np.array(a_norm.multiply(b_norm).sum(axis=1)).ravel()
        return scores.astype(np.float32)


# ─── Feature extraction per pair ─────────────────────────────────────────────

def compute_pair_features(
    s1_row: pd.Series,
    tgt_row: pd.Series,
    name_tfidf: Optional[TFIDFFeatureComputer] = None,
    addr_tfidf: Optional[TFIDFFeatureComputer] = None,
) -> Dict[str, float]:
    """Compute all similarity features for a single (S1, candidate) pair.

    Returns: dict of feature_name → float value.
    """
    feats = {}

    # ── Unpack fields ──────────────────────────────────────────────────────
    s1_name     = str(s1_row.get("norm_name", "") or "")
    s1_name_raw = str(s1_row.get("business_name", "") or "").lower()
    s1_addr     = str(s1_row.get("norm_address", "") or "")
    s1_country  = str(s1_row.get("norm_country", "") or "")
    s1_postal   = str(s1_row.get("postal_code", "") or "")

    t_name     = str(tgt_row.get("norm_name", "") or "")
    t_name_raw = str(tgt_row.get("business_name", "") or "").lower()
    t_addr     = str(tgt_row.get("norm_address", "") or "")
    t_country  = str(tgt_row.get("norm_country", "") or "")
    t_postal   = str(tgt_row.get("postal_code", "") or "")

    s1_name_toks = set(get_name_tokens(s1_name))
    t_name_toks  = set(get_name_tokens(t_name))
    s1_addr_toks = set(get_address_tokens(s1_addr))
    t_addr_toks  = set(get_address_tokens(t_addr))

    # ── Business Name Features ─────────────────────────────────────────────

    # Exact normalized match
    feats["name_exact_match"]       = float(s1_name == t_name and bool(s1_name))

    # Levenshtein similarity (normalized)
    feats["name_levenshtein"]       = levenshtein_sim(s1_name, t_name)

    # Jaro-Winkler
    feats["name_jaro_winkler"]      = jaro_winkler_sim(s1_name, t_name)

    # Token sort ratio (order insensitive)
    feats["name_token_sort"]        = token_sort_ratio(s1_name, t_name)

    # Token set ratio (subset insensitive)
    feats["name_token_set"]         = token_set_ratio(s1_name, t_name)

    # Token Jaccard
    feats["name_jaccard"]           = jaccard_similarity(s1_name_toks, t_name_toks)

    # Character n-gram similarities
    feats["name_char2gram"]         = char_ngram_similarity(s1_name, t_name, 2)
    feats["name_char3gram"]         = char_ngram_similarity(s1_name, t_name, 3)
    feats["name_char4gram"]         = char_ngram_similarity(s1_name, t_name, 4)

    # Prefix similarity
    feats["name_prefix5"]           = prefix_similarity(s1_name, t_name, 5)
    feats["name_prefix3"]           = prefix_similarity(s1_name, t_name, 3)

    # Common token count
    feats["name_common_tokens"]     = float(len(s1_name_toks & t_name_toks))

    # Length difference
    feats["name_len_diff"]          = abs(len(s1_name) - len(t_name)) / (max(len(s1_name), len(t_name)) + 1)

    # Token count difference
    feats["name_token_count_diff"]  = abs(len(s1_name_toks) - len(t_name_toks))

    # Raw name similarity (before normalization)
    feats["name_raw_levenshtein"]   = levenshtein_sim(s1_name_raw, t_name_raw)

    # TF-IDF cosine (name)
    if name_tfidf is not None:
        feats["name_tfidf_cosine"]  = name_tfidf.cosine_sim(s1_name, t_name)
    else:
        feats["name_tfidf_cosine"]  = feats["name_char3gram"]  # fallback

    # ── Address Features ───────────────────────────────────────────────────

    # Exact match
    feats["addr_exact_match"]       = float(s1_addr == t_addr and bool(s1_addr))

    # Levenshtein
    feats["addr_levenshtein"]       = levenshtein_sim(s1_addr, t_addr)

    # Token Jaccard
    feats["addr_jaccard"]           = jaccard_similarity(s1_addr_toks, t_addr_toks)

    # Character n-gram
    feats["addr_char3gram"]         = char_ngram_similarity(s1_addr, t_addr, 3)

    # Token sort
    feats["addr_token_sort"]        = token_sort_ratio(s1_addr, t_addr)

    # Common address tokens
    feats["addr_common_tokens"]     = float(len(s1_addr_toks & t_addr_toks))

    # Length difference
    feats["addr_len_diff"]          = abs(len(s1_addr) - len(t_addr)) / (max(len(s1_addr), len(t_addr)) + 1)

    # Numeric token overlap (building numbers, PIN codes)
    s1_nums = extract_numeric_tokens(s1_addr)
    t_nums  = extract_numeric_tokens(t_addr)
    feats["addr_numeric_overlap"]   = jaccard_similarity(s1_nums, t_nums)
    feats["addr_numeric_exact"]     = float(bool(s1_nums & t_nums))

    # Postal code match
    if s1_postal and t_postal and s1_postal != "None" and t_postal != "None":
        feats["postal_exact_match"] = float(s1_postal == t_postal)
        feats["postal_prefix_match"] = float(s1_postal[:3] == t_postal[:3])
    else:
        feats["postal_exact_match"]  = 0.0
        feats["postal_prefix_match"] = 0.0

    # TF-IDF cosine (address)
    if addr_tfidf is not None:
        feats["addr_tfidf_cosine"]  = addr_tfidf.cosine_sim(s1_addr, t_addr)
    else:
        feats["addr_tfidf_cosine"]  = feats["addr_char3gram"]  # fallback

    # ── Country Features ───────────────────────────────────────────────────

    feats["country_exact_match"]    = float(s1_country == t_country and bool(s1_country))
    feats["country_sim"]            = levenshtein_sim(s1_country, t_country)

    # ── Combined / Cross Features ──────────────────────────────────────────

    # Are name AND address both similar?
    feats["name_and_addr_both_high"] = float(
        feats["name_levenshtein"] > 0.7 and feats["addr_levenshtein"] > 0.5
    )

    # Max name score
    feats["name_max_sim"] = max(
        feats["name_levenshtein"],
        feats["name_token_sort"],
        feats["name_token_set"],
        feats["name_tfidf_cosine"],
    )

    return feats


# ─── Batch feature computation ────────────────────────────────────────────────

def _batch_rapidfuzz(queries: List[str], choices: List[str], scorer) -> np.ndarray:
    """Batch rapidfuzz scorer for lists of string pairs."""
    if HAS_RAPIDFUZZ:
        from rapidfuzz import process as rfprocess
        # Use rapidfuzz cdist for vectorized computation
        try:
            from rapidfuzz.distance import Levenshtein as _Lev
            # Fall back to pair-wise for small batches
            results = np.array([
                scorer(q, c) / 100.0 if hasattr(scorer, '__call__') else scorer(q, c)
                for q, c in zip(queries, choices)
            ], dtype=np.float32)
            return results
        except Exception:
            pass
    return np.array([levenshtein_sim(q, c) for q, c in zip(queries, choices)], dtype=np.float32)


def build_feature_matrix(
    pairs: List[Tuple[str, str]],  # list of (s1_id, target_id)
    s1_dict: Dict,                 # entity_id → row dict/Series
    target_dict: Dict,             # entity_id → row dict/Series
    name_tfidf: Optional[TFIDFFeatureComputer] = None,
    addr_tfidf: Optional[TFIDFFeatureComputer] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute feature matrix for a list of candidate pairs.

    Uses batch processing for string similarity metrics where possible.
    Returns:
        (DataFrame of features, list of feature names)
    """
    if not pairs:
        return pd.DataFrame(), []

    logger.info(f"Computing features for {len(pairs):,} candidate pairs...")

    # Extract all text fields in batch
    s1_names_norm = []
    s1_names_raw  = []
    s1_addrs      = []
    s1_countries  = []
    s1_postals    = []
    t_names_norm  = []
    t_names_raw   = []
    t_addrs       = []
    t_countries   = []
    t_postals     = []
    valid_mask    = []

    for s1_id, tgt_id in pairs:
        s1_row  = s1_dict.get(s1_id)
        tgt_row = target_dict.get(tgt_id)
        if s1_row is None or tgt_row is None:
            valid_mask.append(False)
            s1_names_norm.append(""); s1_names_raw.append(""); s1_addrs.append("")
            s1_countries.append(""); s1_postals.append("")
            t_names_norm.append(""); t_names_raw.append(""); t_addrs.append("")
            t_countries.append(""); t_postals.append("")
            continue
        valid_mask.append(True)
        s1_names_norm.append(str(s1_row.get("norm_name", "") or ""))
        s1_names_raw.append(str(s1_row.get("business_name", "") or "").lower())
        s1_addrs.append(str(s1_row.get("norm_address", "") or ""))
        s1_countries.append(str(s1_row.get("norm_country", "") or ""))
        s1_postals.append(str(s1_row.get("postal_code", "") or ""))
        t_names_norm.append(str(tgt_row.get("norm_name", "") or ""))
        t_names_raw.append(str(tgt_row.get("business_name", "") or "").lower())
        t_addrs.append(str(tgt_row.get("norm_address", "") or ""))
        t_countries.append(str(tgt_row.get("norm_country", "") or ""))
        t_postals.append(str(tgt_row.get("postal_code", "") or ""))

    n = len(pairs)

    # ── Batch string similarity computation ────────────────────────────────
    if HAS_RAPIDFUZZ:
        name_lev   = np.array([rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_jw    = np.array([_JaroWinkler.similarity(a, b) for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_tsort = np.array([rfuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_tset  = np.array([rfuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_partial = np.array([rfuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_wratio  = np.array([rfuzz.WRatio(a, b) / 100.0 for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_raw   = np.array([rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_names_raw, t_names_raw)], dtype=np.float32)
        addr_lev   = np.array([rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
        addr_tsort = np.array([rfuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
        addr_partial = np.array([rfuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
        country_sim = np.array([rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_countries, t_countries)], dtype=np.float32)
    else:
        name_lev   = np.array([levenshtein_sim(a, b) for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_jw    = name_lev.copy()
        name_tsort = np.array([token_sort_ratio(a, b) for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_tset  = np.array([token_set_ratio(a, b) for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
        name_partial = name_lev.copy()
        name_wratio  = name_lev.copy()
        name_raw   = np.array([levenshtein_sim(a, b) for a, b in zip(s1_names_raw, t_names_raw)], dtype=np.float32)
        addr_lev   = np.array([levenshtein_sim(a, b) for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
        addr_tsort = np.array([token_sort_ratio(a, b) for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
        addr_partial = addr_lev.copy()
        country_sim = np.array([levenshtein_sim(a, b) for a, b in zip(s1_countries, t_countries)], dtype=np.float32)

    # ── Token-based Jaccard (vectorized via sets) ─────────────────────────
    def batch_jaccard_name(a_list, b_list):
        results = np.zeros(len(a_list), dtype=np.float32)
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            sa = set(get_name_tokens(a))
            sb = set(get_name_tokens(b))
            results[i] = jaccard_similarity(sa, sb)
        return results

    def batch_jaccard_addr(a_list, b_list):
        results = np.zeros(len(a_list), dtype=np.float32)
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            sa = set(get_address_tokens(a))
            sb = set(get_address_tokens(b))
            results[i] = jaccard_similarity(sa, sb)
        return results

    def batch_common_tokens_name(a_list, b_list):
        return np.array([
            float(len(set(get_name_tokens(a)) & set(get_name_tokens(b))))
            for a, b in zip(a_list, b_list)
        ], dtype=np.float32)

    def batch_common_tokens_addr(a_list, b_list):
        return np.array([
            float(len(set(get_address_tokens(a)) & set(get_address_tokens(b))))
            for a, b in zip(a_list, b_list)
        ], dtype=np.float32)

    def batch_char_ngram(a_list, b_list, n):
        return np.array([char_ngram_similarity(a, b, n) for a, b in zip(a_list, b_list)], dtype=np.float32)

    def batch_prefix(a_list, b_list, n):
        if HAS_RAPIDFUZZ:
            return np.array([rfuzz.ratio(a[:n], b[:n]) / 100.0 for a, b in zip(a_list, b_list)], dtype=np.float32)
        return np.array([levenshtein_sim(a[:n], b[:n]) for a, b in zip(a_list, b_list)], dtype=np.float32)

    def batch_numeric_jaccard(a_list, b_list):
        results = np.zeros(len(a_list), dtype=np.float32)
        has_overlap = np.zeros(len(a_list), dtype=np.float32)
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            sa = extract_numeric_tokens(a)
            sb = extract_numeric_tokens(b)
            results[i] = jaccard_similarity(sa, sb)
            has_overlap[i] = float(bool(sa & sb))
        return results, has_overlap

    name_jaccard = batch_jaccard_name(s1_names_norm, t_names_norm)
    addr_jaccard = batch_jaccard_addr(s1_addrs, t_addrs)
    name_common  = batch_common_tokens_name(s1_names_norm, t_names_norm)
    addr_common  = batch_common_tokens_addr(s1_addrs, t_addrs)
    name_char2   = batch_char_ngram(s1_names_norm, t_names_norm, 2)
    name_char3   = batch_char_ngram(s1_names_norm, t_names_norm, 3)
    name_char4   = batch_char_ngram(s1_names_norm, t_names_norm, 4)
    addr_char3   = batch_char_ngram(s1_addrs, t_addrs, 3)
    name_prefix5 = batch_prefix(s1_names_norm, t_names_norm, 5)
    name_prefix3 = batch_prefix(s1_names_norm, t_names_norm, 3)
    addr_numeric_jac, addr_numeric_exact = batch_numeric_jaccard(s1_addrs, t_addrs)

    # ── Exact matches ─────────────────────────────────────────────────────
    name_exact    = np.array([float(a == b and bool(a)) for a, b in zip(s1_names_norm, t_names_norm)], dtype=np.float32)
    addr_exact    = np.array([float(a == b and bool(a)) for a, b in zip(s1_addrs, t_addrs)], dtype=np.float32)
    country_exact = np.array([float(a == b and bool(a)) for a, b in zip(s1_countries, t_countries)], dtype=np.float32)

    # ── Length differences ────────────────────────────────────────────────
    name_len_diff  = np.array([
        abs(len(a) - len(b)) / (max(len(a), len(b)) + 1)
        for a, b in zip(s1_names_norm, t_names_norm)
    ], dtype=np.float32)
    name_tok_diff  = np.array([
        abs(len(get_name_tokens(a)) - len(get_name_tokens(b)))
        for a, b in zip(s1_names_norm, t_names_norm)
    ], dtype=np.float32)
    addr_len_diff  = np.array([
        abs(len(a) - len(b)) / (max(len(a), len(b)) + 1)
        for a, b in zip(s1_addrs, t_addrs)
    ], dtype=np.float32)

    # ── Postal code ───────────────────────────────────────────────────────
    postal_exact  = np.array([
        float(a == b and bool(a) and a != "None" and b != "None")
        for a, b in zip(s1_postals, t_postals)
    ], dtype=np.float32)
    postal_prefix = np.array([
        float(a[:3] == b[:3] and bool(a) and bool(b) and a != "None" and b != "None")
        for a, b in zip(s1_postals, t_postals)
    ], dtype=np.float32)

    # ── TF-IDF cosine ─────────────────────────────────────────────────────
    if name_tfidf is not None:
        name_tfidf_cos = name_tfidf.batch_cosine_sim(list(zip(s1_names_norm, t_names_norm)))
    else:
        name_tfidf_cos = name_char3.copy()
    if addr_tfidf is not None:
        addr_tfidf_cos = addr_tfidf.batch_cosine_sim(list(zip(s1_addrs, t_addrs)))
    else:
        addr_tfidf_cos = addr_char3.copy()

    name_max_sim = np.maximum(
        np.maximum(name_lev, name_tsort),
        np.maximum(name_tset, name_tfidf_cos)
    )
    name_max_sim = np.maximum(name_max_sim, name_partial)
    both_high    = ((name_lev > 0.7) & (addr_lev > 0.5)).astype(np.float32)
    name_high_addr_partial = ((name_lev > 0.8) & (addr_partial > 0.6)).astype(np.float32)

    # ── First-token match feature ─────────────────────────────────────────
    def first_token(s: str) -> str:
        toks = s.split()
        return toks[0] if toks else ""

    first_tok_match = np.array([
        float(
            bool(first_token(a)) and
            bool(first_token(b)) and
            first_token(a) == first_token(b)
        )
        for a, b in zip(s1_names_norm, t_names_norm)
    ], dtype=np.float32)

    addr_first_tok_match = np.array([
        float(
            bool(first_token(a)) and
            bool(first_token(b)) and
            first_token(a) == first_token(b)
        )
        for a, b in zip(s1_addrs, t_addrs)
    ], dtype=np.float32)

    data = {
        # Name features
        "name_exact_match":          name_exact,
        "name_levenshtein":          name_lev,
        "name_jaro_winkler":         name_jw,
        "name_token_sort":           name_tsort,
        "name_token_set":            name_tset,
        "name_partial_ratio":        name_partial,
        "name_wratio":               name_wratio,
        "name_jaccard":              name_jaccard,
        "name_char2gram":            name_char2,
        "name_char3gram":            name_char3,
        "name_char4gram":            name_char4,
        "name_prefix5":              name_prefix5,
        "name_prefix3":              name_prefix3,
        "name_tfidf_cosine":         name_tfidf_cos,
        "name_common_tokens":        name_common,
        "name_len_diff":             name_len_diff,
        "name_token_count_diff":     name_tok_diff,
        "name_raw_levenshtein":      name_raw,
        "name_first_token_match":    first_tok_match,
        # Address features
        "addr_exact_match":          addr_exact,
        "addr_levenshtein":          addr_lev,
        "addr_jaccard":              addr_jaccard,
        "addr_char3gram":            addr_char3,
        "addr_token_sort":           addr_tsort,
        "addr_partial_ratio":        addr_partial,
        "addr_tfidf_cosine":         addr_tfidf_cos,
        "addr_common_tokens":        addr_common,
        "addr_len_diff":             addr_len_diff,
        "addr_numeric_overlap":      addr_numeric_jac,
        "addr_numeric_exact":        addr_numeric_exact,
        "addr_first_token_match":    addr_first_tok_match,
        "postal_exact_match":        postal_exact,
        "postal_prefix_match":       postal_prefix,
        # Country features
        "country_exact_match":       country_exact,
        "country_sim":               country_sim,
        # Cross features
        "name_and_addr_both_high":   both_high,
        "name_high_addr_partial":    name_high_addr_partial,
        "name_max_sim":              name_max_sim,
    }

    df = pd.DataFrame(data)
    df = df.fillna(0.0)

    feature_names = df.columns.tolist()
    logger.info(f"  → {len(feature_names)} features computed.")
    return df, feature_names


def fit_tfidf_computers(all_names: List[str], all_addrs: List[str]) -> Tuple:
    """Fit TF-IDF computers on all available texts (train + test combined).

    Returns: (name_tfidf, addr_tfidf)
    """
    logger.info("Fitting TF-IDF vectorizers...")

    name_tfidf = TFIDFFeatureComputer(
        analyzer="char_wb", ngram_range=(2, 4),
        max_features=cfg.FEATURES["tfidf_max_features"]
    )
    name_tfidf.fit([n or "" for n in all_names])

    addr_tfidf = TFIDFFeatureComputer(
        analyzer="char_wb", ngram_range=(2, 3),
        max_features=30000
    )
    addr_tfidf.fit([a or "" for a in all_addrs])

    logger.info("TF-IDF vectorizers fitted.")
    return name_tfidf, addr_tfidf
