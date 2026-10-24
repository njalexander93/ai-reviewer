"""Canonical UTF-8 evidence serialization and snapshot hashing shared across layers."""

import hashlib
import json
from typing import Any


def encode(value: Any) -> str:
    """Serialize exact Unicode evidence consistently without ASCII expansion."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(text: str) -> str:
    """Bind persisted work to the exact UTF-8 payload."""
    return hashlib.sha256(text.encode()).hexdigest()
