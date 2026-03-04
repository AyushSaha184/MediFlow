"""
src/agents/medical_rag_agent.py
-------------------------------
Medical Knowledge RAG Agent.
Handles dynamic loading of patient files into a localized FAISS store,
and provides a unified retrieval interface to query both the Patient Store
and the Global Knowledge Store simultaneously.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.base_agent import BaseAgent
from src.models.medical_document import MedicalDocumentSchema
from src.rag.common import build_chunk_id, flatten_record
from src.rag.embedding_service import EmbeddingService
from src.rag.faiss_store import FAISSStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

GLOBAL_STORE_DIR = Path("src/rag/global_store")
PATIENT_DATA_ROOT = Path("data/User")


class MedicalRAGAgent(BaseAgent):
    def __init__(
        self,
        embedder: EmbeddingService,
        global_store_dir: Path = GLOBAL_STORE_DIR,
        patient_data_root: Path = PATIENT_DATA_ROOT,
    ) -> None:
        super().__init__("MedicalRAGAgent")
        self.embedder = embedder
        self.global_store_dir = Path(global_store_dir)
        self.patient_data_root = Path(patient_data_root)
        self.global_store = FAISSStore.load_local(
            str(self.global_store_dir),
            dimension=self.embedder.dimension,
        )
        self._patient_store_cache: Dict[str, FAISSStore] = {}

    def get_patient_store_dir(self, session_id: str) -> Path:
        return self.patient_data_root / session_id / "rag_patient"

    async def ingest_patient_documents(self, session_id: str, documents: List[MedicalDocumentSchema]) -> Dict[str, int]:
        """
        Takes processed documents (with `.chunks`), embeds them, and creates
        a session-local FAISS store under `data/User/<session_id>/rag_patient`.
        """
        session = (session_id or "").strip()
        if not session:
            raise ValueError("session_id is required for patient indexing.")

        store_path = self.get_patient_store_dir(session)
        if store_path.exists():
            shutil.rmtree(store_path)

        patient_store = FAISSStore(
            dimension=self.embedder.dimension,
            required_metadata_keys={"chunk_id", "document_id", "source_file", "origin", "session_id", "text"},
        )

        all_chunks: List[Dict[str, Any]] = []
        chunks_seen = 0
        for doc in documents:
            chunks = doc.chunks or []
            if not chunks:
                continue

            source_file = (
                doc.metadata.get("source_filename")
                or doc.metadata.get("filename")
                or "unknown"
            )

            for chunk_index, chunk in enumerate(chunks):
                chunks_seen += 1
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue

                section = str(chunk.get("section", "GENERAL") or "GENERAL")
                raw_chunk_metadata = chunk.get("metadata")
                chunk_metadata = raw_chunk_metadata if isinstance(raw_chunk_metadata, dict) else {}

                payload = {
                    "chunk_id": build_chunk_id(source_file, section, text),
                    "document_id": doc.document_id,
                    "source_file": source_file,
                    "origin": "patient_store",
                    "session_id": session,
                    "section": section,
                    "text": text,
                    "metadata": {
                        **chunk_metadata,
                        "chunk_index": chunk_index,
                    },
                }
                all_chunks.append(payload)

        if not all_chunks:
            self.logger.warning("rag_ingest_empty", session_id=session, msg="No patient chunks found to embed.")
            self._patient_store_cache.pop(session, None)
            return {"chunks_seen": chunks_seen, "chunks_embedded": 0, "documents_indexed": len(documents)}

        texts = [c["text"] for c in all_chunks]
        self.logger.info("rag_embedding_patient", session_id=session, count=len(texts))
        embeddings = self.embedder.embed_batch(texts)

        chunks_added = patient_store.add(embeddings=embeddings, metadatas=all_chunks, dedupe_by_chunk_id=True)
        patient_store.save_local(str(store_path))
        self._patient_store_cache[session] = patient_store

        for doc in documents:
            if self.name not in doc.processed_by:
                doc.processed_by.append(self.name)

        self.logger.info(
            "rag_patient_ingest_complete",
            session_id=session,
            chunks_seen=chunks_seen,
            chunks_embedded=chunks_added,
            documents_indexed=len(documents),
        )
        return {"chunks_seen": chunks_seen, "chunks_embedded": chunks_added, "documents_indexed": len(documents)}

    def _get_patient_store(self, session_id: str) -> Optional[FAISSStore]:
        if session_id in self._patient_store_cache:
            return self._patient_store_cache[session_id]

        store_path = self.get_patient_store_dir(session_id)
        if not store_path.exists():
            return None

        store = FAISSStore.load_local(str(store_path), dimension=self.embedder.dimension)
        self._patient_store_cache[session_id] = store
        return store

    @staticmethod
    def _format_retrieval_hit(distance: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized = flatten_record(metadata)
        section = str(normalized.get("section") or "GENERAL")
        source_file = str(normalized.get("source_file") or "unknown")
        text = str(normalized.get("text") or "")
        chunk_id = str(normalized.get("chunk_id") or build_chunk_id(source_file, section, text))
        origin = str(normalized.get("origin") or "unknown")
        document_id = str(normalized.get("document_id") or "")
        session_id = normalized.get("session_id")
        payload_meta = normalized.get("metadata")
        metadata_payload = payload_meta if isinstance(payload_meta, dict) else {}

        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_file": source_file,
            "origin": origin,
            "section": section,
            "text": text,
            "session_id": session_id,
            "distance": float(distance),
            "l2_distance": float(distance),
            "metadata": metadata_payload,
        }

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k_patient: int = 5,
        top_k_global: int = 5,
        top_k_total: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Search session patient store and global knowledge store, then merge by L2 distance.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty.")
        if top_k_patient <= 0 or top_k_global <= 0 or top_k_total <= 0:
            raise ValueError("top-k values must be positive integers.")

        self.logger.info(
            "rag_query",
            query=query,
            session_id=session_id,
            top_k_patient=top_k_patient,
            top_k_global=top_k_global,
            top_k_total=top_k_total,
        )

        q_vec = self.embedder.embed_text(query)

        merged_results: List[tuple[float, Dict[str, Any]]] = []

        if session_id:
            patient_store = self._get_patient_store(session_id)
            if patient_store and patient_store.index.ntotal > 0:
                patient_results = patient_store.search(
                    q_vec,
                    k=top_k_patient,
                    metadata_filter={"session_id": session_id},
                )
                merged_results.extend(patient_results)
            else:
                self.logger.info("rag_patient_store_missing", session_id=session_id)

        if self.global_store.index.ntotal > 0:
            global_results = self.global_store.search(q_vec, k=top_k_global)
            merged_results.extend(global_results)

        if not merged_results:
            self.logger.info("rag_query_no_results", query=query, session_id=session_id)
            return []

        merged_results.sort(key=lambda item: item[0])

        final_chunks: List[Dict[str, Any]] = []
        seen_chunk_ids = set()
        for distance, metadata in merged_results:
            formatted = self._format_retrieval_hit(distance, metadata)
            if formatted["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(formatted["chunk_id"])
            final_chunks.append(formatted)
            if len(final_chunks) >= top_k_total:
                break

        return final_chunks

    def cleanup_session(self, session_id: str) -> bool:
        """
        Delete only the RAG patient store directory for a session.
        """
        session = (session_id or "").strip()
        if not session:
            raise ValueError("session_id is required for cleanup.")

        store_path = self.get_patient_store_dir(session)
        deleted = store_path.exists()
        if deleted:
            shutil.rmtree(store_path, ignore_errors=True)

        self._patient_store_cache.pop(session, None)
        self.logger.info("rag_session_cleanup", session_id=session, deleted=deleted)
        return deleted

    async def run(self, documents: List[MedicalDocumentSchema], session_id: str) -> List[MedicalDocumentSchema]:
        """
        Agent entrypoint. Ingests all passed documents into the session-local patient store.
        """
        await self.ingest_patient_documents(session_id=session_id, documents=documents)
        return documents
