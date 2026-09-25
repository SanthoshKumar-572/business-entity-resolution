"""
normalize.py - Text normalization module for Business Entity Resolution.
Exposes clean normalization functions for names, addresses, and country labels.
"""

from .normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
    extract_numeric_tokens,
    get_name_tokens,
    get_name_token_set,
    get_address_tokens,
    get_name_prefix_key,
    unicode_normalize,
    normalize_record,
    normalize_dataframe,
)

__all__ = [
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "extract_postal_code",
    "extract_numeric_tokens",
    "get_name_tokens",
    "get_name_token_set",
    "get_address_tokens",
    "get_name_prefix_key",
    "unicode_normalize",
    "normalize_record",
    "normalize_dataframe",
]
