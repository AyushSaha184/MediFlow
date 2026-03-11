"""
src/rag/guideline_ingestor.py
-----------------------------
Loads curated guideline sources from a manifest file and local files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in (".md", ".txt"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            return "\n".join(page.get_text() for page in doc)
        except Exception as exc:
            logger.warning("guideline_pdf_read_failed", file=file_path.name, error=str(exc))
            return ""

    if suffix == ".csv":
        try:
            import pandas as pd

            df = pd.read_csv(file_path)
            lines = []
            for _, row in df.iterrows():
                lines.append("  |  ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip()))
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("guideline_csv_read_failed", file=file_path.name, error=str(exc))
            return ""

    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            logger.warning("guideline_docx_read_failed", file=file_path.name, error=str(exc))
            return ""

    return ""


def load_guideline_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        logger.warning("guideline_manifest_missing", path=str(path))
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("guideline_manifest_parse_failed", path=str(path), error=str(exc))
        return []

    entries = payload.get("guidelines", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def load_guideline_documents(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Convert curated manifest entries into ingestion-ready documents.
    Supports local files in v1 for deterministic ingestion.
    """
    rows = load_guideline_manifest(manifest_path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        local_path = str(row.get("local_path") or "").strip()
        if not local_path:
            continue
        file_path = Path(local_path)
        if not file_path.exists():
            logger.warning("guideline_source_missing", local_path=local_path)
            continue
        text = _read_file(file_path)
        if not text.strip():
            continue
        out.append(
            {
                "document_id": str(row.get("document_id") or f"guideline::{file_path.stem}"),
                "source_file": str(row.get("source_file") or file_path.name),
                "title": str(row.get("title") or file_path.stem),
                "text": text,
                "issuer": str(row.get("issuer") or "unknown"),
                "source_type": str(row.get("source_type") or "guideline"),
                "source_tier": str(row.get("source_tier") or "authoritative"),
                "evidence_level": str(row.get("evidence_level") or "A"),
                "specialty": str(row.get("specialty") or "general"),
                "jurisdiction": str(row.get("jurisdiction") or "global"),
                "published_at": str(row.get("published_at") or ""),
                "last_reviewed_at": str(row.get("last_reviewed_at") or ""),
                "pmid": str(row.get("pmid") or ""),
                "doi": str(row.get("doi") or ""),
                "topic": str(row.get("topic") or ""),
            }
        )
    logger.info("guideline_documents_loaded", count=len(out), manifest_path=manifest_path)
    return out
