"""Deterministic hashing helpers shared by the cache, config loader, and run
metadata modules.

A single canonical JSON serialization is used everywhere a hash is derived
from a Python object, so the same logical content always produces the same
hash regardless of dict key insertion order or float formatting quirks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize `obj` to a stable JSON string.

    Keys are sorted and separators are fixed so semantically identical inputs
    always produce byte-identical output.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_object(obj: Any) -> str:
    """Hash an arbitrary JSON-serializable object via its canonical form."""
    return sha256_hex(canonical_json(obj))


def hash_text(text: str) -> str:
    return sha256_hex(text)
