"""
test_output.py - Unit tests for output formatting and validation.
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import pandas as pd
from business_entity_resolution.src.output_writer import (
    write_matching_results,
    write_candidate_pairs,
    validate_outputs_are_consistent,
)


class TestOutputFormat:

    def _write_and_read(self, predictions, s1_ids, writer_fn, col_name):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            tmp_path = f.name

        try:
            writer_fn(predictions, s1_ids, path=tmp_path)
            df = pd.read_csv(tmp_path, sep="\t", dtype=str).fillna("")
            return df
        finally:
            os.unlink(tmp_path)

    def test_matching_results_one_row_per_s1(self):
        s1_ids = ["S1-001", "S1-002", "S1-003"]
        predictions = {
            "S1-001": {"S2-001", "S3-001"},
            "S1-002": set(),
        }
        df = self._write_and_read(predictions, s1_ids, write_matching_results, "matched_entity_ids")
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    def test_matching_results_has_correct_header(self):
        s1_ids = ["S1-001"]
        predictions = {"S1-001": {"S2-001"}}
        df = self._write_and_read(predictions, s1_ids, write_matching_results, "matched_entity_ids")
        assert "source1_entity_id" in df.columns
        assert "matched_entity_ids" in df.columns

    def test_empty_prediction_is_empty_string(self):
        s1_ids = ["S1-001"]
        predictions = {}  # no predictions
        df = self._write_and_read(predictions, s1_ids, write_matching_results, "matched_entity_ids")
        row = df[df["source1_entity_id"] == "S1-001"].iloc[0]
        assert row["matched_entity_ids"] == "", f"Expected empty, got '{row['matched_entity_ids']}'"

    def test_no_s1_ids_in_matched(self):
        """S1 IDs should be filtered out of matched_entity_ids."""
        s1_ids = ["S1-001"]
        # Include invalid S1 ID in predictions (should be filtered)
        predictions = {"S1-001": {"S2-001", "S1-999"}}
        df = self._write_and_read(predictions, s1_ids, write_matching_results, "matched_entity_ids")
        row = df[df["source1_entity_id"] == "S1-001"].iloc[0]
        matched = row["matched_entity_ids"]
        assert "S1-" not in matched, f"S1 ID should not appear in matched: {matched}"

    def test_no_duplicate_ids_in_list(self):
        s1_ids = ["S1-001"]
        predictions = {"S1-001": {"S2-001", "S2-002"}}  # set prevents duplicates
        df = self._write_and_read(predictions, s1_ids, write_matching_results, "matched_entity_ids")
        row = df[df["source1_entity_id"] == "S1-001"].iloc[0]
        matched_list = row["matched_entity_ids"].split(",")
        assert len(matched_list) == len(set(matched_list)), "Duplicate IDs found"

    def test_candidate_pairs_one_row_per_s1(self):
        s1_ids = ["S1-001", "S1-002"]
        candidates = {
            "S1-001": {"S2-001", "S3-001"},
        }
        df = self._write_and_read(candidates, s1_ids, write_candidate_pairs, "candidate_entity_ids")
        assert len(df) == 2

    def test_tab_separator(self):
        """Output must be TAB-separated."""
        s1_ids = ["S1-001"]
        predictions = {"S1-001": {"S2-001"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
        try:
            write_matching_results(predictions, s1_ids, path=tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                first_line = f.readline()
            assert "\t" in first_line, "Output should be TAB-separated"
        finally:
            os.unlink(tmp_path)

    def test_utf8_encoding(self):
        """Output should be valid UTF-8."""
        s1_ids = ["S1-001"]
        predictions = {"S1-001": set()}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
        try:
            write_matching_results(predictions, s1_ids, path=tmp_path)
            # Should not raise
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            assert "source1_entity_id" in content
        finally:
            os.unlink(tmp_path)


class TestOutputConsistency:

    def test_matches_subset_of_candidates(self):
        predictions = {
            "S1-001": {"S2-001"},
            "S1-002": {"S3-001"},
        }
        candidates = {
            "S1-001": {"S2-001", "S2-002"},
            "S1-002": {"S3-001"},
        }
        assert validate_outputs_are_consistent(predictions, candidates) is True

    def test_match_not_in_candidates_returns_false(self):
        predictions = {"S1-001": {"S2-999"}}  # S2-999 not in candidates
        candidates = {"S1-001": {"S2-001"}}
        assert validate_outputs_are_consistent(predictions, candidates) is False

    def test_empty_predictions_always_consistent(self):
        predictions = {"S1-001": set()}
        candidates = {"S1-001": {"S2-001"}}
        assert validate_outputs_are_consistent(predictions, candidates) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
