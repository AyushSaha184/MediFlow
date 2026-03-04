"""
Shared helpers for the Phase 4 RAG components.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict


def build_chunk_id(source_file: str, section: str, text: str) -> str:
    """
    Build a stable chunk identifier from source + section + text.
    """
    payload = f"{source_file.strip().lower()}|{section.strip().lower()}|{text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """
    Return UTC timestamp in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a stored metadata record into a flat retrieval payload.

    Older records may store important keys inside `record["metadata"]`.
    """
    normalized = dict(record)
    nested_meta = normalized.get("metadata")
    if not isinstance(nested_meta, dict):
        nested_meta = {}
        normalized["metadata"] = nested_meta

    for field in ("source_file", "origin", "chunk_id", "document_id", "session_id", "section", "text"):
        if field not in normalized and field in nested_meta:
            normalized[field] = nested_meta[field]

    return normalized
