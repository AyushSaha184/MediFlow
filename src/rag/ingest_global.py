"""
src/rag/ingest_global.py
------------------------
CLI Script to parse Markdown/Text medical guidelines from `data/knowledge_base/` 
and embed them into the persistent `src/rag/global_store/` FAISS index.

Run this script manually whenever new literature is added to the system.
Usage: python -m src.rag.ingest_global
"""

from pathlib import Path
from typing import Dict, List

from src.rag.common import build_chunk_id, flatten_record, utc_now_iso
from src.rag.embedding_service import EmbeddingService
from src.rag.pgvector_store import PGVectorStore, TABLE_KNOWLEDGE
from src.utils.logger import get_logger
from src.services.chunking_service import ChunkingService
from src.core.config import settings

logger = get_logger(__name__)

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
GLOBAL_STORE_DIR = "src/rag/global_store"


def _iter_knowledge_files(kb_path: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.md", "*.txt"):
        files.extend(sorted(kb_path.glob(ext)))
    return files


def _derive_existing_chunk_ids(store: PGVectorStore) -> set[str]:
    chunk_ids = set(store.existing_chunk_ids())
    for raw_record in store.metadata_store:
        record = flatten_record(raw_record)
        chunk_id = record.get("chunk_id")
        if chunk_id:
            chunk_ids.add(str(chunk_id))
            continue

        source_file = str(record.get("source_file") or "unknown")
        section = str(record.get("section") or "GENERAL")
        text = str(record.get("text") or "")
        if text:
            chunk_ids.add(build_chunk_id(source_file, section, text))
    return chunk_ids


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
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

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
