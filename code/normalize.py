"""
normalize.py — Shared normalization utilities for business entity resolution.

Used by all pipeline stages: candidate generation, feature engineering, inference.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal suffix expansions (abbreviation → canonical form)
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\binc\b": "inc",
    r"\bincorp\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bcorp\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bltd\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bpvt\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bco\b": "co",
    r"\bcompany\b": "co",
    r"\bgroup\b": "grp",
    r"\bgrp\b": "grp",
    r"\benterprises\b": "ent",
    r"\benterprise\b": "ent",
    r"\bservices\b": "svcs",
    r"\bservice\b": "svc",
    r"\bindustries\b": "inds",
    r"\bindustry\b": "ind",
    r"\bsolutions\b": "sol",
    r"\bassociates\b": "assoc",
    r"\bassociation\b": "assoc",
    r"\btrading\b": "trd",
    r"\btraders\b": "trd",
    r"\bconsulting\b": "consult",
    r"\bconsultants\b": "consult",
    r"\bholdings\b": "hldg",
    r"\bholding\b": "hldg",
    r"\bventures\b": "vent",
    r"\bventure\b": "vent",
    r"\binternational\b": "intl",
    r"\bnational\b": "natl",
    r"\bglobal\b": "global",
}

# Address abbreviations
ADDR_ABBREVS = {
    r"\bstreet\b": "st",
    r"\bst\b": "st",
    r"\broad\b": "rd",
    r"\brd\b": "rd",
    r"\bavenue\b": "ave",
    r"\bave\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bblvd\b": "blvd",
    r"\bdrive\b": "dr",
    r"\bdr\b": "dr",
    r"\blane\b": "ln",
    r"\bln\b": "ln",
    r"\bcourt\b": "ct",
    r"\bct\b": "ct",
    r"\bcircle\b": "cir",
    r"\bcir\b": "cir",
    r"\bplace\b": "pl",
    r"\bpl\b": "pl",
    r"\bsuite\b": "ste",
    r"\bste\b": "ste",
    r"\bapartment\b": "apt",
    r"\bapt\b": "apt",
    r"\bunit\b": "unit",
    r"\bfloor\b": "fl",
    r"\bbuilding\b": "bldg",
    r"\bparkway\b": "pkwy",
    r"\bpkwy\b": "pkwy",
    r"\bexpressway\b": "expy",
    r"\bhighway\b": "hwy",
    r"\bhwy\b": "hwy",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bnortheast\b": "ne",
    r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se",
    r"\bsouthwest\b": "sw",
    r"\bpost office box\b": "po box",
    r"\bp\.o\. box\b": "po box",
    r"\bpo box\b": "po box",
}

_COMPILED_LEGAL = [(re.compile(pat), rep) for pat, rep in LEGAL_SUFFIXES.items()]
_COMPILED_ADDR = [(re.compile(pat), rep) for pat, rep in ADDR_ABBREVS.items()]


def unicode_normalize(text: str) -> str:
    """Normalize unicode to ASCII-compatible form, preserving devanagari etc as-is."""
    if not isinstance(text, str):
        return ""
    # NFKC normalization (decomposes ligatures, compatibility chars)
    text = unicodedata.normalize("NFKC", text)
    return text


def normalize_name(name: str) -> str:
    """
    Normalize a business name for comparison/blocking.

    Steps:
    1. Unicode normalize
    2. Lowercase
    3. Remove special characters (keep alphanumeric + spaces)
    4. Expand & → and
    5. Apply legal suffix abbreviations
    6. Strip extra whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    text = unicode_normalize(name)
    text = text.lower().strip()

    # Expand & → and
    text = text.replace("&", " and ")

    # Remove punctuation except spaces
    text = re.sub(r"[^a-z0-9\u0900-\u097f\u00c0-\u024f\s]", " ", text)

    # Apply legal suffix normalization
    for pat, rep in _COMPILED_LEGAL:
        text = pat.sub(rep, text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(addr: str) -> str:
    """
    Normalize a business address for comparison/blocking.

    Steps:
    1. Unicode normalize
    2. Lowercase
    3. Remove punctuation
    4. Apply address abbreviations
    5. Strip extra whitespace
    """
    if not isinstance(addr, str) or not addr.strip():
        return ""

    text = unicode_normalize(addr)
    text = text.lower().strip()

    # Remove punctuation except spaces, digits, letters
    text = re.sub(r"[^a-z0-9\u0900-\u097f\u00c0-\u024f\s/]", " ", text)

    # Apply address abbreviations
    for pat, rep in _COMPILED_ADDR:
        text = pat.sub(rep, text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(country: str) -> str:
    """Normalize country label to lowercase stripped."""
    if not isinstance(country, str):
        return ""
    return country.lower().strip()


def get_name_tokens(normalized_name: str) -> set:
    """Return non-empty tokens from a normalized name."""
    if not normalized_name:
        return set()
    return set(t for t in normalized_name.split() if len(t) > 1)


def get_address_tokens(normalized_addr: str) -> set:
    """Return non-empty tokens from a normalized address."""
    if not normalized_addr:
        return set()
    return set(t for t in normalized_addr.split() if len(t) > 1)


def get_name_prefix(normalized_name: str, length: int = 4) -> str:
    """Get the first `length` chars of first token for blocking prefix key."""
    if not normalized_name:
        return ""
    first_token = normalized_name.split()[0] if normalized_name.split() else ""
    return first_token[:length]


def extract_numbers(text: str) -> list:
    """Extract all numeric strings from text."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\d+", text)


def get_blocking_keys(norm_name: str, norm_addr: str, country: str) -> list:
    """
    Generate multiple blocking keys for a record.

    Returns a list of (key_type, key_value) tuples.
    These are used to build inverted indexes for candidate generation.

    Strategy: Compound keys that are specific enough to avoid combinatorial
    explosion while still capturing all likely matches (high recall).
    Trigram keys removed to limit false-positive explosion.
    """
    keys = []

    tokens = norm_name.split() if norm_name else []
    # Filter stopwords from candidate token list
    stop = {"the", "and", "of", "in", "at", "to", "for", "a", "an"}
    non_trivial = [t for t in tokens if len(t) >= 3 and t not in stop]

    # ── Name-based keys ──────────────────────────────────────────────────────

    # Key 1: first significant token (≥4 chars) + country
    if non_trivial:
        t0 = non_trivial[0]
        if len(t0) >= 4:
            keys.append(("name_tok0_country", f"{t0[:6]}_{country}"))

    # Key 2: sorted bigram of first two significant tokens + country
    if len(non_trivial) >= 2:
        bigram = "_".join(sorted([non_trivial[0][:6], non_trivial[1][:6]]))
        keys.append(("name_bigram_country", f"{bigram}_{country}"))

    # Key 3: prefix compound (5+4 chars of first two tokens) + country
    if len(non_trivial) >= 2:
        compound = f"{non_trivial[0][:5]}_{non_trivial[1][:4]}_{country}"
        keys.append(("name_compound_country", compound))

    # ── Address-based keys ────────────────────────────────────────────────────

    addr_nums = extract_numbers(norm_addr)
    addr_tokens = [t for t in norm_addr.split() if len(t) > 1] if norm_addr else []
    significant_addr = [t for t in addr_tokens if not t.isdigit() and len(t) >= 3]

    # Key 4: house number + first name token + country (very precise)
    if addr_nums and non_trivial:
        keys.append(("addr_num_name_tok0", f"{addr_nums[0]}_{non_trivial[0][:5]}_{country}"))

    # Key 5: first two significant address tokens + country
    if len(significant_addr) >= 2:
        addr_key = f"{significant_addr[0][:8]}_{significant_addr[1][:6]}_{country}"
        keys.append(("addr_sig2_country", addr_key))

    # Key 6: house number + first significant address token + country
    if addr_nums and significant_addr:
        keys.append(("addr_num_sig1_country",
                      f"{addr_nums[0]}_{significant_addr[0][:6]}_{country}"))

    return keys
