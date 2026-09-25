"""
test_features.py - Unit tests for the feature engineering module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import pandas as pd
from business_entity_resolution.src.features import (
    levenshtein_sim,
    jaro_winkler_sim,
    jaccard_similarity,
    char_ngram_similarity,
    token_sort_ratio,
    compute_pair_features,
)


class TestStringSimilarityMetrics:

    def test_levenshtein_identical(self):
        assert levenshtein_sim("hello", "hello") == pytest.approx(1.0)

    def test_levenshtein_empty_both(self):
        assert levenshtein_sim("", "") == pytest.approx(1.0)

    def test_levenshtein_one_empty(self):
        assert levenshtein_sim("hello", "") == pytest.approx(0.0)

    def test_levenshtein_range(self):
        score = levenshtein_sim("acme corp", "acme corporation")
        assert 0.0 <= score <= 1.0

    def test_jaro_identical(self):
        assert jaro_winkler_sim("hello", "hello") == pytest.approx(1.0)

    def test_jaro_range(self):
        score = jaro_winkler_sim("acme", "acme corp")
        assert 0.0 <= score <= 1.0

    def test_jaccard_identical(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_jaccard_empty_both(self):
        assert jaccard_similarity(set(), set()) == pytest.approx(1.0)

    def test_char_ngram_identical(self):
        score = char_ngram_similarity("hello", "hello", n=3)
        assert score == pytest.approx(1.0)

    def test_char_ngram_range(self):
        score = char_ngram_similarity("hello world", "helo world", n=3)
        assert 0.0 <= score <= 1.0

    def test_token_sort_order_insensitive(self):
        score1 = token_sort_ratio("acme corp inc", "inc acme corp")
        # Token sort should give high score for reordered tokens
        assert score1 > 0.8


class TestPairFeatures:

    def _make_row(self, name="acme corp", address="123 main st", country="us", postal="10001"):
        return pd.Series({
            "entity_id":          "S1-001",
            "business_name":      name,
            "business_address":   address,
            "country":            country,
            "norm_name":          name.lower(),
            "norm_address":       address.lower(),
            "norm_country":       country.lower(),
            "postal_code":        postal,
        })

    def test_identical_records(self):
        row = self._make_row()
        feats = compute_pair_features(row, row)
        assert feats["name_exact_match"] == 1.0
        assert feats["addr_exact_match"] == 1.0
        assert feats["country_exact_match"] == 1.0

    def test_feature_range_all_values_in_0_1(self):
        row_a = self._make_row("acme corp", "123 main st", "us", "10001")
        row_b = self._make_row("acme corporation", "123 main street", "us", "10001")
        feats = compute_pair_features(row_a, row_b)

        # All similarity scores should be in [0, 1]
        similarity_feats = [k for k in feats if not k.endswith("_diff") and not k.endswith("_count") and not k.endswith("_tokens")]
        for k in similarity_feats:
            val = feats[k]
            assert 0.0 <= val <= 1.0, f"Feature '{k}' = {val} out of range"

    def test_country_mismatch(self):
        row_a = self._make_row(country="us")
        row_b = self._make_row(country="india")
        feats = compute_pair_features(row_a, row_b)
        assert feats["country_exact_match"] == 0.0

    def test_postal_match(self):
        row_a = self._make_row(postal="411001")
        row_b = self._make_row(postal="411001")
        feats = compute_pair_features(row_a, row_b)
        assert feats["postal_exact_match"] == 1.0

    def test_postal_mismatch(self):
        row_a = self._make_row(postal="411001")
        row_b = self._make_row(postal="560001")
        feats = compute_pair_features(row_a, row_b)
        assert feats["postal_exact_match"] == 0.0

    def test_feature_count(self):
        row = self._make_row()
        feats = compute_pair_features(row, row)
        # Should have many features
        assert len(feats) >= 20, f"Only {len(feats)} features generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
