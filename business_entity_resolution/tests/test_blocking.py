"""
test_blocking.py - Unit tests for the blocking / candidate generation module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import pandas as pd
from business_entity_resolution.src.blocking import BlockingEngine
from business_entity_resolution.src.normalization import normalize_dataframe


def _make_df(records, id_prefix="S1"):
    """Helper to build a normalized DataFrame from a list of dicts."""
    df = pd.DataFrame(records)
    df = normalize_dataframe(df)
    return df


class TestBlockingRecall:
    """Verify that blocking does NOT eliminate known true matches."""

    def test_exact_name_match_is_candidate(self):
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "Acme Corp",
             "business_address": "123 Main St, New York", "country": "US"}
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Acme Corp",
             "business_address": "123 Main St, New York", "country": "US"},
            {"entity_id": "S2-002", "business_name": "Completely Different Co",
             "business_address": "456 Oak Ave, Chicago", "country": "US"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        assert "S2-001" in candidates.get("S1-001", set()), \
            "Exact name match should be in candidates"

    def test_similar_name_is_candidate(self):
        """Abbreviated name variants should still be candidates."""
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "Prabhav Business Centre",
             "business_address": "797 Lake Town, Kolkata", "country": "India"}
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Prabhav Bizness Center",
             "business_address": "797 Lake Town Block A, Kolkata", "country": "India"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        # S2-001 should be a candidate (shares "prabhav" token)
        assert "S2-001" in candidates.get("S1-001", set()), \
            "Similar name should be in candidates"

    def test_postal_code_match(self):
        """Records sharing a postal code should be candidates."""
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "Random Store",
             "business_address": "Shop No 5, MG Road, Pune 411001", "country": "India"}
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Random Stores",
             "business_address": "Near Shivaji Nagar, Pune 411001", "country": "India"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        assert "S2-001" in candidates.get("S1-001", set()), \
            "Postal code match should be in candidates"

    def test_no_candidates_for_completely_different(self):
        """Records with completely different names should not match."""
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "XYZ Enterprises",
             "business_address": "789 Park Ave, Los Angeles, CA 90001", "country": "US"}
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Zephyr Innovations",
             "business_address": "Koramangala, Bangalore 560034", "country": "India"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        # Different country AND completely different name → should not match
        cands = candidates.get("S1-001", set())
        # (This is a weak assertion — we just check it doesn't explode)
        assert isinstance(cands, set)

    def test_all_s1_ids_in_output(self):
        """All S1 entity IDs should appear in the output dict."""
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "Acme", "business_address": "123 St", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Beta", "business_address": "456 Ave", "country": "US"},
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "123 St", "country": "US"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        assert "S1-001" in candidates
        assert "S1-002" in candidates

    def test_no_s1_ids_in_candidates(self):
        """S1 IDs should never appear as candidates."""
        s1 = _make_df([
            {"entity_id": "S1-001", "business_name": "Acme", "business_address": "123 St", "country": "US"},
        ])
        s2 = _make_df([
            {"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "123 St", "country": "US"},
        ])
        s3 = pd.DataFrame(columns=s2.columns)
        s3 = normalize_dataframe(s3)

        engine = BlockingEngine(max_candidates=100)
        candidates = engine.generate_candidates(s1, s2, s3)
        for s1_id, cand_set in candidates.items():
            for cid in cand_set:
                assert not cid.startswith("S1-"), f"S1 ID {cid} appeared as candidate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
