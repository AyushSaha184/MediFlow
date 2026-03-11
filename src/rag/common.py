"""
Shared helpers for the Phase 4 RAG components.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

KB_METADATA_KEYS = (
    "canonical_id",
    "source_tier",
    "source_type",
    "evidence_level",
    "published_at",
    "last_reviewed_at",
    "superseded",
    "specialty",
    "jurisdiction",
    "issuer",
    "pmid",
    "doi",
)


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

    for field in (
        "source_file",
        "origin",
        "chunk_id",
        "document_id",
        "session_id",
        "section",
        "text",
        *KB_METADATA_KEYS,
    ):
        if field not in normalized and field in nested_meta:
            normalized[field] = nested_meta[field]

    return normalized


def normalize_text_for_hash(value: str) -> str:
    """
    Normalize free text for stable hashing and canonical IDs.
    """
    compact = re.sub(r"\s+", " ", (value or "").strip().lower())
    return compact


def canonical_topic_slug(value: str) -> str:
    """
    Convert topic/title text into a short stable slug.
    """
    cleaned = re.sub(r"[^a-z0-9\s-]", "", normalize_text_for_hash(value))
    return re.sub(r"[\s-]+", "-", cleaned).strip("-")


def first_non_empty(values: Iterable[str], default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default
