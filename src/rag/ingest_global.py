"""
src/rag/ingest_global.py
------------------------
CLI Script to parse medical literature from PubMed + local knowledge files and embed
them into the persistent pgvector `mediflow_knowledge` table.
Run this script manually whenever new literature is added to the system.
Usage: python -m src.rag.ingest_global
"""

from pathlib import Path
from typing import Any, Dict, List

from src.rag.common import build_chunk_id, utc_now_iso
from src.rag.canonicalize import (
    SOURCE_TIER_PRIORITY,
    make_guideline_identity,
    normalize_source_metadata,
    parse_iso_datetime_or_min,
)
from src.rag.embedding_service import EmbeddingService
from src.rag.pgvector_store import PGVectorStore, TABLE_KNOWLEDGE
from src.rag.pubmed_ingestor import PubMedIngestConfig, build_pubmed_query, fetch_pubmed_records
from src.utils.logger import get_logger
from src.services.chunking_service import ChunkingService
from src.core.config import settings

logger = get_logger(__name__)

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
GLOBAL_STORE_DIR = "src/rag/global_store"
_SUPPORTED_EXTS = ("*.md", "*.txt", "*.pdf", "*.csv", "*.docx")


def _iter_knowledge_files(kb_path: Path) -> List[Path]:
    files: List[Path] = []
    for ext in _SUPPORTED_EXTS:
        files.extend(sorted(kb_path.glob(ext)))
    return files


def _read_file(file_path: Path) -> str:
    """Extract plain text from supported local file types."""
    suffix = file_path.suffix.lower()

    if suffix in (".md", ".txt"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            return "\n".join(page.get_text() for page in doc)
        except Exception as exc:
            logger.warning("ingest_pdf_read_failed", file=file_path.name, error=str(exc))
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
            logger.warning("ingest_csv_read_failed", file=file_path.name, error=str(exc))
            return ""

    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            logger.warning("ingest_docx_read_failed", file=file_path.name, error=str(exc))
            return ""

    return ""


def _derive_existing_chunk_ids(store: PGVectorStore) -> set[str]:
    return set(store.existing_chunk_ids())


def _iter_pubmed_documents() -> List[Dict[str, Any]]:
    if not settings.kb_enable_pubmed:
        logger.warning("pubmed_ingest_disabled", msg="Set KB_ENABLE_PUBMED=true to ingest from PubMed.")
        return []
    email = settings.kb_pubmed_email.strip()
    if not email:
        logger.warning("pubmed_ingest_disabled_missing_email")
        return []

    queries_raw = [q.strip() for q in settings.kb_pubmed_mesh_queries.split(";") if q.strip()]
    if not queries_raw:
        logger.warning("pubmed_ingest_disabled_missing_queries")
        return []

    config = PubMedIngestConfig(
        email=email,
        api_key=settings.kb_pubmed_api_key,
        max_results=settings.kb_pubmed_max_results,
    )

    rows: List[Dict[str, Any]] = []
    for query_term in queries_raw:
        query = build_pubmed_query(topic=query_term, mesh_term=query_term)
        try:
            rows.extend(fetch_pubmed_records(query=query, config=config))
        except Exception as exc:
            logger.warning("pubmed_ingest_query_failed", query=query_term, error=str(exc))
    return rows


def _iter_local_documents(kb_path: Path) -> List[Dict[str, Any]]:
    files = _iter_knowledge_files(kb_path)
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        logger.info("global_ingestion_processing_local_file", file=file_path.name)
        content = _read_file(file_path)
        if not content.strip():
            logger.warning("global_ingestion_empty_file", file=file_path.name)
            continue
        rows.append(
            {
                "document_id": f"local::{file_path.name}",
                "source_file": file_path.name,
                "title": file_path.stem,
                "text": content,
                "issuer": "local_knowledge_base",
                "source_type": "reference",
                "source_tier": "reference",
                "evidence_level": "C",
                "specialty": "general",
                "jurisdiction": "global",
                "published_at": "",
                "last_reviewed_at": "",
                "pmid": "",
                "doi": "",
                "topic": file_path.stem,
            }
        )
    return rows


def _mark_superseded_versions(store: PGVectorStore, metadata_rows: List[Dict[str, Any]]) -> int:
    """
    Mark older rows as superseded when a newer version of same canonical_id is ingested.
    """
    updates = 0
    seen: Dict[str, Dict[str, Any]] = {}
    for row in metadata_rows:
        canonical_id = str(row.get("canonical_id") or "").strip()
        if not canonical_id:
            continue
        incumbent = seen.get(canonical_id)
        if incumbent is None:
            seen[canonical_id] = row
            continue
        incumbent_dt = parse_iso_datetime_or_min(str(incumbent.get("published_at") or ""))
        current_dt = parse_iso_datetime_or_min(str(row.get("published_at") or ""))
        if current_dt > incumbent_dt:
            seen[canonical_id] = row

    for canonical_id, winner in seen.items():
        winner_doc_id = str(winner.get("document_id") or "")
        with store._get_cursor() as cur:
            cur.execute(
                f"""
                UPDATE {TABLE_KNOWLEDGE}
                SET metadata = jsonb_set(metadata, '{{superseded}}', 'true', true)
                WHERE namespace = %s
                  AND COALESCE(metadata->>'canonical_id', '') = %s
                  AND COALESCE(metadata->>'document_id', '') <> %s
                """,
                (store.namespace, canonical_id, winner_doc_id),
            )
            updates += max(cur.rowcount, 0)
    return updates


def _tier_distance_multiplier(source_tier: str) -> float:
    priority = SOURCE_TIER_PRIORITY.get(source_tier, 1)
    # Better sources receive lower distance multiplier.
    if priority >= 3:
        return 0.95
    if priority == 2:
        return 1.0
    return 1.05


def _is_near_duplicate(
    store: PGVectorStore,
    vector_row,
    source_meta: Dict[str, Any],
    threshold: float,
) -> bool:
    """
    Skip adding chunks that are nearly identical to existing chunks in same specialty/topic.
    Uses L2 distance over normalized vectors.
    """
    specialty = str(source_meta.get("specialty") or "general")
    topic = str(source_meta.get("topic") or "")
    # Query top-k and compare only same specialty/topic if available.
    existing = store.search(query_vector=vector_row, k=5)
    for distance, metadata in existing:
        existing_specialty = str(metadata.get("specialty") or "")
        existing_topic = str(metadata.get("topic") or "")
        if existing_specialty and existing_specialty != specialty:
            continue
        if topic and existing_topic and existing_topic != topic:
            continue
        if float(distance) <= threshold:
            return True
    return False


def run_ingestion() -> Dict[str, int]:
    logger.info("global_ingestion_started", source="pubmed_and_local")
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        logger.warning("knowledge_base_missing", path=str(kb_path), msg="Continuing with PubMed-only source.")

    chunker = ChunkingService(target_chunk_size=1000, overlap=150)
    embedder = EmbeddingService(
        provider=settings.rag_embedding_provider,
        model_name=settings.rag_embedding_model_name,
        fallback_dimension=settings.rag_embedding_fallback_dimension,
        local_files_only=settings.rag_embedding_local_files_only,
        nvidia_api_url=settings.rag_embedding_nvidia_api_url,
        nvidia_api_key=settings.rag_embedding_nvidia_api_key,
        nvidia_truncate=settings.rag_embedding_nvidia_truncate,
        request_timeout_seconds=settings.rag_embedding_request_timeout_seconds,
        nvidia_max_batch_size=settings.rag_embedding_nvidia_max_batch_size,
    )

    store = PGVectorStore.load_local(GLOBAL_STORE_DIR, dimension=embedder.dimension, table_name=TABLE_KNOWLEDGE)
    existing_chunk_ids = _derive_existing_chunk_ids(store)
    local_docs = _iter_local_documents(kb_path) if kb_path.exists() else []
    pubmed_docs = _iter_pubmed_documents()
    source_docs = local_docs + pubmed_docs

    chunks_to_add = []
    normalized_source_rows: List[Dict[str, Any]] = []
    chunks_total = 0
    chunks_skipped = 0
    near_duplicates = 0

    for source in source_docs:
        content = str(source.get("text") or "").strip()
        if not content:
            continue

        source_meta = normalize_source_metadata(source)
        normalized_source_rows.append(
            {
                **source_meta,
                "document_id": str(source.get("document_id") or ""),
            }
        )
        chunks = chunker.chunk_document(content)
        for chunk in chunks:
            chunks_total += 1
            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text:
                chunks_skipped += 1
                continue

            section = str(chunk.get("section", "GENERAL") or "GENERAL")
            chunk_id = build_chunk_id(str(source.get("source_file") or source_meta["title"]), section, chunk_text)
            if chunk_id in existing_chunk_ids:
                chunks_skipped += 1
                continue

            existing_chunk_ids.add(chunk_id)

            chunk_meta = chunk.get("metadata")
            if not isinstance(chunk_meta, dict):
                chunk_meta = {}

            chunks_to_add.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": str(source.get("document_id") or f"global::{source_meta['title']}"),
                    "source_file": str(source.get("source_file") or "unknown"),
                    "origin": "global_store",
                    "section": section,
                    "text": chunk_text,
                    "ingested_at": utc_now_iso(),
                    "embedding_model": embedder.model_name,
                    "canonical_id": source_meta["canonical_id"],
                    "source_tier": source_meta["source_tier"],
                    "source_type": source_meta["source_type"],
                    "evidence_level": source_meta["evidence_level"],
                    "published_at": source_meta["published_at"],
                    "last_reviewed_at": source_meta["last_reviewed_at"],
                    "superseded": source_meta["superseded"],
                    "specialty": source_meta["specialty"],
                    "jurisdiction": source_meta["jurisdiction"],
                    "issuer": source_meta["issuer"],
                    "pmid": source_meta["pmid"],
                    "doi": source_meta["doi"],
                    "topic": source_meta["topic"],
                    "metadata": chunk_meta,
                }
            )

    if chunks_to_add:
        logger.info("global_ingestion_embedding_batch", chunk_count=len(chunks_to_add))
        embeddings = embedder.embed_batch([chunk["text"] for chunk in chunks_to_add])

        filtered_chunks = []
        filtered_vectors = []
        for idx, chunk in enumerate(chunks_to_add):
            vector = embeddings[idx]
            if _is_near_duplicate(
                store=store,
                vector_row=vector,
                source_meta=chunk,
                threshold=float(settings.kb_near_duplicate_threshold),
            ):
                near_duplicates += 1
                continue
            chunk["distance_multiplier"] = _tier_distance_multiplier(str(chunk.get("source_tier") or "reference"))
            filtered_chunks.append(chunk)
            filtered_vectors.append(vector)

        if filtered_chunks:
            import numpy as np

            chunks_added = store.add(
                embeddings=np.asarray(filtered_vectors, dtype=np.float32),
                metadatas=filtered_chunks,
                dedupe_by_chunk_id=True,
            )
        else:
            chunks_added = 0
    else:
        chunks_added = 0

    superseded_rows = _mark_superseded_versions(store=store, metadata_rows=normalized_source_rows)

    store.save_local(GLOBAL_STORE_DIR)

    summary = {
        "files_scanned": len(source_docs),
        "chunks_total": chunks_total,
        "chunks_added": chunks_added,
        "chunks_skipped": chunks_skipped,
        "near_duplicates_skipped": near_duplicates,
        "superseded_rows_marked": superseded_rows,
    }
    logger.info("global_ingestion_complete", **summary, store_total=store.index.ntotal)
    return summary


def main():
    summary = run_ingestion()
    logger.info("global_ingestion_summary", **summary)

if __name__ == "__main__":
    main()
