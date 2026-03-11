"""
src/rag/canonicalize.py
-----------------------
Canonical identity and metadata helpers for global medical knowledge ingestion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from src.rag.common import canonical_topic_slug, first_non_empty, normalize_text_for_hash


SOURCE_TIER_PRIORITY = {
    "authoritative": 3,
    "evidence": 2,
    "reference": 1,
}


def _safe_year(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return ""


def build_canonical_id(
    *,
    title: str,
    issuer: str,
    topic: str = "",
    published_at: str = "",
) -> str:
    """
    Create a stable canonical ID shared across mirrored/duplicate sources.
    """
    normalized_title = normalize_text_for_hash(title)
    normalized_issuer = normalize_text_for_hash(issuer)
    normalized_topic = canonical_topic_slug(topic or title)
    year = _safe_year(published_at)
    payload = f"{normalized_issuer}|{normalized_topic}|{normalized_title}|{year}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_source_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize source-level metadata to the KB contract.
    """
    title = first_non_empty((raw.get("title"), raw.get("source_file"), raw.get("document_id")))
    issuer = first_non_empty((raw.get("issuer"), raw.get("publisher"), raw.get("journal"), "unknown"))
    source_type = first_non_empty((raw.get("source_type"), "reference"))
    source_tier = first_non_empty((raw.get("source_tier"), "reference"))
    topic = first_non_empty((raw.get("topic"), raw.get("specialty"), title))
    published_at = first_non_empty((raw.get("published_at"), raw.get("publication_date"), ""))
    evidence_level = first_non_empty((raw.get("evidence_level"), "C"))
    specialty = first_non_empty((raw.get("specialty"), "general"))
    jurisdiction = first_non_empty((raw.get("jurisdiction"), "global"))
    canonical_id = first_non_empty((raw.get("canonical_id"), ""))
    if not canonical_id:
        canonical_id = build_canonical_id(
            title=title,
            issuer=issuer,
            topic=topic,
            published_at=published_at,
        )
    return {
        "title": title,
        "issuer": issuer,
        "source_type": source_type,
        "source_tier": source_tier,
        "topic": topic,
        "published_at": published_at,
        "evidence_level": evidence_level,
        "specialty": specialty,
        "jurisdiction": jurisdiction,
        "canonical_id": canonical_id,
        "pmid": str(raw.get("pmid") or ""),
        "doi": str(raw.get("doi") or ""),
        "superseded": bool(raw.get("superseded", False)),
        "last_reviewed_at": str(raw.get("last_reviewed_at") or ""),
    }


@dataclass(frozen=True)
class GuidelineIdentity:
    canonical_id: str
    issuer: str
    topic: str
    published_at: str


def make_guideline_identity(metadata: Dict[str, Any]) -> GuidelineIdentity:
    normalized = normalize_source_metadata(metadata)
    return GuidelineIdentity(
        canonical_id=normalized["canonical_id"],
        issuer=normalized["issuer"],
        topic=normalized["topic"],
        published_at=normalized["published_at"],
    )


def parse_iso_datetime_or_min(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.min
