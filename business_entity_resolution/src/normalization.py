"""
normalization.py - Text normalization for business names and addresses.

All normalization is done locally — no external APIs or geocoding services.
Both original and normalized values are preserved for feature engineering.
"""

import re
import unicodedata
import logging
from typing import Optional

from . import config as cfg

logger = logging.getLogger(__name__)


# ─── Compile patterns once ────────────────────────────────────────────────────

def _compile_patterns(mapping: dict) -> list:
    """Compile a pattern dict into a list of (compiled_re, replacement) tuples."""
    return [
        (re.compile(pat, re.IGNORECASE | re.UNICODE), repl)
        for pat, repl in mapping.items()
    ]


_LEGAL_SUFFIX_PATTERNS = _compile_patterns(cfg.NORMALIZATION["legal_suffixes"])
_NAME_ABBREV_PATTERNS  = _compile_patterns(cfg.NORMALIZATION["name_abbrevs"])
_ADDR_ABBREV_PATTERNS  = _compile_patterns(cfg.NORMALIZATION["addr_abbrevs"])

# Extra patterns not in config
_PUNCT_RE      = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE   = re.compile(r"\s+")
_DIGITS_RE     = re.compile(r"\d+")

# Pattern to strip common noise prefixes like << or --
_NOISE_PREFIX  = re.compile(r"^[^\w]+")


# ─── Shared utilities ─────────────────────────────────────────────────────────

def unicode_normalize(text: str) -> str:
    """Normalize Unicode to NFC form and strip accents (NFKD decompose → ASCII)."""
    # First NFC
    text = unicodedata.normalize("NFC", text)
    # Try to transliterate accented chars by NFKD + encode/decode
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    return ascii_bytes.decode("ascii")


def _apply_patterns(text: str, patterns: list) -> str:
    """Apply a list of (regex, replacement) pairs sequentially."""
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text


# ─── Business Name Normalization ──────────────────────────────────────────────

def normalize_business_name(name: str) -> str:
    """Normalize a business name string.

    Steps:
    1. Strip whitespace
    2. Unicode normalization
    3. Lowercase
    4. Remove noise prefix characters
    5. Normalize & → and
    6. Expand common abbreviations
    7. Normalize legal suffixes
    8. Remove punctuation
    9. Collapse whitespace
    """
    if not name or not isinstance(name, str):
        return ""

    text = name.strip()
    text = unicode_normalize(text)
    text = text.lower()

    # Remove noise prefix characters (<< -- etc)
    text = _NOISE_PREFIX.sub("", text)

    # Apply name abbreviation expansions (includes & → and)
    text = _apply_patterns(text, _NAME_ABBREV_PATTERNS)

    # Apply legal suffix normalization
    text = _apply_patterns(text, _LEGAL_SUFFIX_PATTERNS)

    # Remove punctuation (keep alphanumeric and spaces)
    text = _PUNCT_RE.sub(" ", text)

    # Collapse multiple whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def get_name_tokens(normalized_name: str) -> list:
    """Return list of meaningful tokens from a normalized name."""
    tokens = normalized_name.split()
    return [t for t in tokens if len(t) >= cfg.BLOCKING["min_token_len"]]


def get_name_token_set(normalized_name: str) -> frozenset:
    """Return frozenset of meaningful tokens from a normalized name."""
    return frozenset(get_name_tokens(normalized_name))


# ─── Name prefix key (first 3 chars of first meaningful token) ───────────────

def get_name_prefix_key(normalized_name: str, n: int = 4) -> str:
    """Get first n chars of normalized name (for blocking key)."""
    text = normalized_name.strip()
    if len(text) >= n:
        return text[:n]
    return text


# ─── Address Normalization ────────────────────────────────────────────────────

def normalize_address(address: str) -> str:
    """Normalize a business address string.

    Steps:
    1. Strip whitespace
    2. Unicode normalization
    3. Lowercase
    4. Expand address abbreviations
    5. Remove excess punctuation (but keep numbers)
    6. Collapse whitespace
    """
    if not address or not isinstance(address, str):
        return ""

    text = address.strip()
    text = unicode_normalize(text)
    text = text.lower()

    # Apply address abbreviation expansions
    text = _apply_patterns(text, _ADDR_ABBREV_PATTERNS)

    # Remove punctuation but preserve digits and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def extract_postal_code(address: str) -> Optional[str]:
    """Extract PIN/ZIP code from address string.

    Looks for:
    - Indian PIN codes: 6 consecutive digits
    - US ZIP codes: 5 digits or 5+4 with hyphen
    - French postal codes: 5 digits
    """
    if not address:
        return None

    # Indian PIN: exactly 6 digits (not surrounded by more digits)
    pin_match = re.search(r"(?<!\d)(\d{6})(?!\d)", address)
    if pin_match:
        return pin_match.group(1)

    # US ZIP: 5 digits optionally followed by -4
    zip_match = re.search(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)", address)
    if zip_match:
        return zip_match.group(1)

    return None


def extract_numeric_tokens(address: str) -> set:
    """Extract all numeric substrings from an address (building numbers, etc.)."""
    return set(re.findall(r"\d+", address))


def get_address_tokens(normalized_address: str) -> list:
    """Return list of meaningful tokens from a normalized address."""
    tokens = normalized_address.split()
    return [t for t in tokens if len(t) >= cfg.BLOCKING["min_token_len"]]


# ─── Country Normalization ────────────────────────────────────────────────────

_COUNTRY_ALIASES = {
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.a": "us",
    "u.s": "us",
    "america": "us",
    "india": "india",
    "ind": "india",
    "bharat": "india",
    "france": "france",
    "fr": "france",
    "uk": "uk",
    "united kingdom": "uk",
    "great britain": "uk",
    "england": "uk",
    "canada": "canada",
    "ca": "canada",
    "australia": "australia",
    "au": "australia",
    "germany": "germany",
    "de": "germany",
    "deutschland": "germany",
    "china": "china",
    "cn": "china",
}


def normalize_country(country: str) -> str:
    """Normalize a country label to a canonical lowercase form.

    Uses an open-set alias table; unknown countries pass through lowercased.
    Never filters or discards any country.
    """
    if not country or not isinstance(country, str):
        return ""

    c = country.strip().lower()
    c = re.sub(r"[^\w\s]", "", c).strip()
    return _COUNTRY_ALIASES.get(c, c)


# ─── Combined record normalization ────────────────────────────────────────────

def normalize_record(row: dict) -> dict:
    """Normalize all text fields in a record dict.

    Returns a new dict with additional _norm_ fields alongside the originals.
    """
    return {
        # Original fields
        "entity_id":          row.get("entity_id", ""),
        "business_name":      row.get("business_name", ""),
        "business_address":   row.get("business_address", ""),
        "country":            row.get("country", ""),
        # Normalized fields
        "norm_name":          normalize_business_name(row.get("business_name", "")),
        "norm_address":       normalize_address(row.get("business_address", "")),
        "norm_country":       normalize_country(row.get("country", "")),
        "postal_code":        extract_postal_code(row.get("business_address", "")),
    }


def normalize_dataframe(df):
    """Add normalized columns to a DataFrame in-place.

    Adds: norm_name, norm_address, norm_country, postal_code
    """
    import pandas as pd
    logger.info(f"Normalizing {len(df):,} records...")

    df = df.copy()
    df["norm_name"]    = df["business_name"].apply(normalize_business_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)
    df["postal_code"]  = df["business_address"].apply(extract_postal_code)

    logger.info("  Normalization complete.")
    return df
