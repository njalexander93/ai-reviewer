"""Canonical UTF-8 evidence serialization and snapshot hashing shared across layers."""

import hashlib
import json
from typing import Any


def encode(value: Any) -> str:
    """Return compact JSON for value with literal Unicode and no optional whitespace.

    Value must be JSON serializable. Mapping insertion order is preserved, not sorted,
    so callers establish deterministic order before encoding. TypeError/ValueError
    propagate for unsupported objects or circular containers. No I/O occurs.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(text: str) -> str:
    """Return the hexadecimal SHA-256 of text encoded as UTF-8.

    Hashes exact text, including whitespace/order; does not canonicalize JSON or prove
    that the evidence is trusted. UnicodeEncodeError propagates for invalid surrogates.
    """
    return hashlib.sha256(text.encode()).hexdigest()
