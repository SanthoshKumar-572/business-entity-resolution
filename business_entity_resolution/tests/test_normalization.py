"""
test_normalization.py - Unit tests for the normalization module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
)


class TestBusinessNameNormalization:

    def test_lowercase(self):
        assert normalize_business_name("ACME CORP") == "acme corp"

    def test_punctuation_removed(self):
        result = normalize_business_name("A.B.C. Ltd.")
        assert "." not in result

    def test_ampersand_to_and(self):
        result = normalize_business_name("Smith & Jones")
        assert "and" in result
        assert "&" not in result

    def test_pvt_limited_normalization(self):
        r1 = normalize_business_name("Pvt Ltd")
        r2 = normalize_business_name("Private Limited")
        r3 = normalize_business_name("Pvt. Ltd.")
        assert r1 == r2 == r3

    def test_inc_vs_incorporated(self):
        r1 = normalize_business_name("Tech Inc")
        r2 = normalize_business_name("Tech Incorporated")
        assert r1 == r2

    def test_whitespace_collapse(self):
        result = normalize_business_name("  Acme   Corp  ")
        assert result == "acme corp"

    def test_empty_string(self):
        assert normalize_business_name("") == ""

    def test_none_handling(self):
        assert normalize_business_name(None) == ""

    def test_preserves_meaningful_tokens(self):
        result = normalize_business_name("McDonald's Restaurant")
        assert "mcdonald" in result
        assert "restaurant" in result

    def test_unicode_normalization(self):
        result = normalize_business_name("Café Bistro")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_legal_suffix_consistency(self):
        variations = [
            "Acme Limited",
            "Acme Ltd",
            "Acme Ltd.",
        ]
        results = [normalize_business_name(v) for v in variations]
        # All should normalize to same form
        assert len(set(results)) == 1, f"Not all normalize the same: {results}"


class TestAddressNormalization:

    def test_lowercase(self):
        assert normalize_address("123 MAIN ST") == "123 main street"

    def test_street_abbreviation(self):
        r1 = normalize_address("123 Main St")
        r2 = normalize_address("123 Main Street")
        assert r1 == r2

    def test_road_abbreviation(self):
        r1 = normalize_address("45 MG Rd Bangalore")
        r2 = normalize_address("45 MG Road Bangalore")
        assert r1 == r2

    def test_whitespace_collapse(self):
        result = normalize_address("  123   Main   St  ")
        assert "  " not in result

    def test_empty_string(self):
        assert normalize_address("") == ""

    def test_preserves_numbers(self):
        result = normalize_address("Plot No 42, Sector 18, Noida")
        assert "42" in result
        assert "18" in result


class TestPostalCodeExtraction:

    def test_us_zip(self):
        assert extract_postal_code("123 Main St, New York, NY 10001") == "10001"

    def test_indian_pin(self):
        assert extract_postal_code("Koregaon Park, Pune 411001, Maharashtra") == "411001"

    def test_no_postal_code(self):
        assert extract_postal_code("Near SBI ATM, MG Road, Bangalore") is None

    def test_empty_string(self):
        assert extract_postal_code("") is None


class TestCountryNormalization:

    def test_us_variants(self):
        assert normalize_country("US") == "us"
        assert normalize_country("USA") == "us"
        assert normalize_country("United States") == "us"

    def test_india_variants(self):
        assert normalize_country("India") == "india"
        assert normalize_country("IND") == "india"

    def test_france(self):
        assert normalize_country("France") == "france"
        assert normalize_country("FR") == "france"

    def test_unknown_country_passes_through(self):
        # Should not fail; unknown countries pass through lowercased
        result = normalize_country("NewCountry")
        assert result == "newcountry"

    def test_empty(self):
        assert normalize_country("") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
