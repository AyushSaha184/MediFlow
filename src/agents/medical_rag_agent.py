"""
src/agents/medical_rag_agent.py
-------------------------------
Medical Knowledge RAG Agent.
Handles dynamic loading of patient files into a localized FAISS store,
and provides a unified retrieval interface to query both the Patient Store
and the Global Knowledge Store simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.base_agent import BaseAgent
from src.models.medical_document import MedicalDocumentSchema
from src.rag.common import build_chunk_id, flatten_record
from src.rag.embedding_service import EmbeddingService
from src.rag.pgvector_store import PGVectorStore, TABLE_KNOWLEDGE, TABLE_PATIENT
from src.rag.crag_graph import build_crag_graph
from src.services.llm_service import LLMService
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
        db_url: Optional[str] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        super().__init__("MedicalRAGAgent")
        self.embedder = embedder
        self.global_store_dir = Path(global_store_dir)
        self.patient_data_root = Path(patient_data_root)
        from src.core.config import settings
        self._db_url = db_url or settings.pgvector_database_url
        self.global_store = PGVectorStore.load_local(
            str(self.global_store_dir),
            dimension=self.embedder.dimension,
            table_name=TABLE_KNOWLEDGE,
            db_url=self._db_url,
        )
        self._patient_store_cache: Dict[str, PGVectorStore] = {}

        # Build the CRAG graph if an LLM service is provided.
        # Without an LLM, the agent falls back to direct (non-corrective) retrieval.
        self._crag_graph = None
        if llm_service is not None:
            self._crag_graph = build_crag_graph(
                embedder=self.embedder,
                global_store=self.global_store,
                get_patient_store_fn=self._get_patient_store,
                llm_service=llm_service,
            )
            self.logger.info("crag_graph_enabled", msg="CRAG StateGraph active for retrieval.")
        else:
            self.logger.info("crag_graph_disabled", msg="No LLM service provided — using direct retrieval.")

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
        patient_store = PGVectorStore.load_local(
            str(store_path),
            dimension=self.embedder.dimension,
            table_name=TABLE_PATIENT,
            required_metadata_keys={"chunk_id", "document_id", "source_file", "origin", "session_id", "text"},
            db_url=self._db_url,
        )
        patient_store.delete_all()  # Clear any stale data for this session

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
                        "document_timestamp": doc.document_timestamp,
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
        # save_local is a no-op for pgvector; data is already persisted in PostgreSQL
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

    def _get_patient_store(self, session_id: str) -> Optional[PGVectorStore]:
        if session_id in self._patient_store_cache:
            return self._patient_store_cache[session_id]

        store_path = self.get_patient_store_dir(session_id)
        store = PGVectorStore.load_local(
            str(store_path),
            dimension=self.embedder.dimension,
            table_name=TABLE_PATIENT,
            db_url=self._db_url,
        )
        if store._count() == 0:
            return None

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

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k_patient: int = 5,
        top_k_global: int = 5,
        top_k_total: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using the CRAG StateGraph when available,
        falling back to direct retrieval if no LLM service was provided.

        Returns the list of formatted chunk dicts.  When CRAG is active and
        retrieval quality is poor after all retries, the chunks are still
        returned but each hit carries ``metadata["low_confidence"] = True``
        so DiagnosticAgent can factor that in.
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
            crag_active=self._crag_graph is not None,
        )

        # ── CRAG path ────────────────────────────────────────────────────────
        if self._crag_graph is not None:
            initial_state = {
                "query": query,
                "session_id": session_id,
                "top_k_patient": top_k_patient,
                "top_k_global": top_k_global,
                "top_k_total": top_k_total,
                "rewrite_count": 0,
                "raw_hits": [],
                "graded_hits": [],
                "rejected_hits": [],
                "results": [],
                "low_confidence": False,
            }
            final_state = await self._crag_graph.ainvoke(initial_state)
            results = final_state.get("results", [])
            low_confidence = final_state.get("low_confidence", False)
            rewrite_count = final_state.get("rewrite_count", 0)

            if low_confidence:
                self.logger.warning(
                    "rag_low_confidence",
                    query=query,
                    session_id=session_id,
                    rewrite_count=rewrite_count,
                    msg="Passing weak context to DiagnosticAgent with low_confidence flag.",
                )
                for hit in results:
                    hit.setdefault("metadata", {})["low_confidence"] = True

            self.logger.info(
                "rag_crag_complete",
                results=len(results),
                low_confidence=low_confidence,
                rewrite_count=rewrite_count,
            )
            return results

        # ── Direct retrieval fallback (no LLM grading) ───────────────────────
        from datetime import datetime
        q_vec = self.embedder.embed_text(query)
        merged: List[tuple[float, Dict[str, Any]]] = []

        if session_id:
            patient_store = self._get_patient_store(session_id)
            if patient_store and patient_store.index.ntotal > 0:
                merged.extend(
                    patient_store.search(q_vec, k=top_k_patient,
                                         metadata_filter={"session_id": session_id})
                )
            else:
                self.logger.info("rag_patient_store_missing", session_id=session_id)

        if self.global_store.index.ntotal > 0:
            merged.extend(self.global_store.search(q_vec, k=top_k_global))

        if not merged:
            self.logger.info("rag_query_no_results", query=query, session_id=session_id)
            return []

        weighted: List[tuple[float, Dict[str, Any]]] = []
        for distance, metadata in merged:
            ts_str = metadata.get("metadata", {}).get("document_timestamp")
            penalty = 1.0
            if ts_str and ts_str != "Historical":
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
                    days_old = (now - dt).days
                    if days_old > 0:
                        penalty = 1.0 + min(0.5, (days_old / 30.0) * 0.05)
                except Exception:
                    penalty = 1.2
            weighted.append((distance * penalty, metadata))

        weighted.sort(key=lambda x: x[0])

        final_chunks: List[Dict[str, Any]] = []
        seen: set = set()
        for distance, metadata in weighted:
            fmt = self._format_retrieval_hit(distance, metadata)
            if fmt["chunk_id"] in seen:
                continue
            seen.add(fmt["chunk_id"])
            final_chunks.append(fmt)
            if len(final_chunks) >= top_k_total:
                break

        return final_chunks

    def cleanup_session(self, session_id: str) -> bool:
        """
        Delete the pgvector rows for the session's patient store namespace.
        """
        session = (session_id or "").strip()
        if not session:
            raise ValueError("session_id is required for cleanup.")

        store_path = self.get_patient_store_dir(session)
        cached = self._patient_store_cache.pop(session, None)
        if cached is not None:
            deleted_count = cached.delete_all()
        else:
            store = PGVectorStore.load_local(
                str(store_path), dimension=self.embedder.dimension, db_url=self._db_url
            )
            deleted_count = store.delete_all()

        deleted = deleted_count > 0
        self.logger.info("rag_session_cleanup", session_id=session, deleted=deleted)
        return deleted

    async def run(self, documents: List[MedicalDocumentSchema], session_id: str) -> List[MedicalDocumentSchema]:
        """
        Agent entrypoint. Ingests all passed documents into the session-local patient store.
        """
        await self.ingest_patient_documents(session_id=session_id, documents=documents)
        return documents
