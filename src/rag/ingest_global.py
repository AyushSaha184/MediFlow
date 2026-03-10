"""
src/rag/ingest_global.py
------------------------
CLI Script to parse medical guidelines from `data/knowledge_base/` and embed
them into the persistent pgvector `mediflow_knowledge` table.

Supported formats: .md  .txt  .pdf  .csv  .docx

Run this script manually whenever new literature is added to the system.
Usage: python -m src.rag.ingest_global
"""

from pathlib import Path
from typing import Dict, List

from src.rag.common import build_chunk_id, utc_now_iso
from src.rag.embedding_service import EmbeddingService
from src.rag.pgvector_store import PGVectorStore, TABLE_KNOWLEDGE
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
    """Extract plain text from any supported file type."""
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
            # Convert each row to a readable sentence-like string
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


def run_ingestion() -> Dict[str, int]:
    logger.info("global_ingestion_started", knowledge_base_dir=KNOWLEDGE_BASE_DIR)

    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        logger.error("knowledge_base_missing", path=str(kb_path))
        return {"files_scanned": 0, "chunks_total": 0, "chunks_added": 0, "chunks_skipped": 0}

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
    files = _iter_knowledge_files(kb_path)

    chunks_to_add = []
    chunks_total = 0
    chunks_skipped = 0

    for file_path in files:
        logger.info("global_ingestion_processing_file", file=file_path.name)
        content = _read_file(file_path)
        if not content.strip():
            logger.warning("global_ingestion_empty_file", file=file_path.name)
            continue

        chunks = chunker.chunk_document(content)
        for chunk in chunks:
            chunks_total += 1
            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text:
                chunks_skipped += 1
                continue

            section = str(chunk.get("section", "GENERAL") or "GENERAL")
            chunk_id = build_chunk_id(file_path.name, section, chunk_text)
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
                    "document_id": f"global::{file_path.name}",
                    "source_file": file_path.name,
                    "origin": "global_store",
                    "section": section,
                    "text": chunk_text,
                    "ingested_at": utc_now_iso(),
                    "embedding_model": embedder.model_name,
                    "metadata": chunk_meta,
                }
            )

    if chunks_to_add:
        logger.info("global_ingestion_embedding_batch", chunk_count=len(chunks_to_add))
        embeddings = embedder.embed_batch([chunk["text"] for chunk in chunks_to_add])
        chunks_added = store.add(embeddings=embeddings, metadatas=chunks_to_add, dedupe_by_chunk_id=True)
    else:
        chunks_added = 0

    store.save_local(GLOBAL_STORE_DIR)

    summary = {
        "files_scanned": len(files),
        "chunks_total": chunks_total,
        "chunks_added": chunks_added,
        "chunks_skipped": chunks_skipped,
    }
    logger.info("global_ingestion_complete", **summary, store_total=store.index.ntotal)
    return summary


def main():
    summary = run_ingestion()
    logger.info("global_ingestion_summary", **summary)

if __name__ == "__main__":
    main()
