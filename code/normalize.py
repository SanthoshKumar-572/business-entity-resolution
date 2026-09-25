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

_LEGAL_LOOKUP = {k.replace(r"\b", ""): v for k, v in LEGAL_SUFFIXES.items()}
_LEGAL_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(_LEGAL_LOOKUP.keys(), key=len, reverse=True)) + r")\b")

_ADDR_LOOKUP = {k.replace(r"\b", ""): v for k, v in ADDR_ABBREVS.items()}
_ADDR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(_ADDR_LOOKUP.keys(), key=len, reverse=True)) + r")\b")


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
    text = re.sub(r"[^a-z0-9\u0900-\u0d7f\u00c0-\u024f\s]", " ", text)

    # Apply legal suffix normalization (single pass)
    text = _LEGAL_PATTERN.sub(lambda m: _LEGAL_LOOKUP[m.group(0)], text)

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
    text = re.sub(r"[^a-z0-9\u0900-\u0d7f\u00c0-\u024f\s/]", " ", text)

    # Apply address abbreviations (single pass)
    text = _ADDR_PATTERN.sub(lambda m: _ADDR_LOOKUP[m.group(0)], text)

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


LEGAL_STOPWORDS = {
    "the", "and", "of", "in", "at", "to", "for", "a", "an", "dba", "com",
    "pvt", "ltd", "inc", "corp", "llc", "llp", "co", "grp", "ent", "sol",
    "inds", "ind", "assoc", "trd", "vent", "intl", "natl", "global", "svcs", "svc"
}

ADDR_STOPWORDS = {
    "flat", "door", "no", "plot", "house", "shop", "floor", "fl", "unit",
    "ste", "apt", "bldg", "room", "first", "second", "third", "near", "opp",
    "behind", "st", "rd", "ave", "dr", "ln", "road", "street"
}


def get_blocking_keys(norm_name: str, norm_addr: str, country: str) -> list:
    """
    Generate multiple blocking keys for a record.

    Returns a list of (key_type, key_value) tuples.
    These are used to build inverted indexes for candidate generation.

    Strategy: Compound keys that are specific enough to avoid combinatorial
    explosion while still capturing all likely matches (high recall).
    """
    keys = []

    tokens = norm_name.split() if norm_name else []
    clean_tokens = [t for t in tokens if len(t) >= 3 and t not in LEGAL_STOPWORDS]
    if not clean_tokens:
        clean_tokens = [t for t in tokens if len(t) >= 3]

    # ── Name-based keys ──────────────────────────────────────────────────────
    if clean_tokens:
        t0 = clean_tokens[0]
        if len(t0) >= 4:
            keys.append(("name_tok0_country", f"{t0[:6]}_{country}"))

    if len(clean_tokens) >= 2:
        bigram = "_".join(sorted([clean_tokens[0][:6], clean_tokens[1][:6]]))
        keys.append(("name_bigram_country", f"{bigram}_{country}"))

        compound = f"{clean_tokens[0][:5]}_{clean_tokens[1][:4]}_{country}"
        keys.append(("name_compound_country", compound))

    # Compacted name for concatenated words (e.g., 'high tech' vs 'hightech')
    if clean_tokens:
        compact = "".join(clean_tokens[:2])[:8]
        if len(compact) >= 5:
            keys.append(("name_compact_country", f"{compact}_{country}"))

    # ── Address-based keys ────────────────────────────────────────────────────
    addr_nums = extract_numbers(norm_addr)
    addr_tokens = [t for t in norm_addr.split() if len(t) > 1] if norm_addr else []
    clean_addr = [t for t in addr_tokens if not t.isdigit() and len(t) >= 3 and t not in ADDR_STOPWORDS]
    if not clean_addr:
        clean_addr = [t for t in addr_tokens if not t.isdigit() and len(t) >= 3]

    # Key 5: house number + first clean name token + country (very precise)
    if addr_nums and clean_tokens:
        keys.append(("addr_num_name_tok0", f"{addr_nums[0]}_{clean_tokens[0][:5]}_{country}"))

    # Key 6: first two significant address tokens + country
    if len(clean_addr) >= 2:
        keys.append(("addr_sig2_country", f"{clean_addr[0][:8]}_{clean_addr[1][:6]}_{country}"))

    # Key 7: house number + first significant address token + country
    if addr_nums and clean_addr:
        keys.append(("addr_num_sig1_country", f"{addr_nums[0]}_{clean_addr[0][:6]}_{country}"))

    return keys

