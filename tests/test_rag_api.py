"""
Endpoint-level tests for Phase 4 RAG routes.
"""

from __future__ import annotations

import pytest

import src.api.main as api_main
from src.models.medical_document import DocumentType, MedicalDocumentSchema
from src.models.rag_models import RAGIndexPatientRequest, RAGRetrieveRequest


class FakeRAGAgent:
    def __init__(self) -> None:
        self.index_calls = []
        self.cleanup_calls = []

    async def ingest_patient_documents(self, session_id: str, documents):
        self.index_calls.append((session_id, len(documents)))
        return {
            "chunks_seen": 3,
            "chunks_embedded": 2,
            "documents_indexed": len(documents),
        }

    def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        top_k_patient: int = 5,
        top_k_global: int = 5,
        top_k_total: int = 8,
    ):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_file": "patient.pdf",
                "origin": "patient_store",
                "section": "HPI",
                "text": "Patient has chest pain.",
                "session_id": session_id,
                "distance": 0.12,
                "l2_distance": 0.12,
                "metadata": {"note": "test"},
            }
        ]

    def cleanup_session(self, session_id: str) -> bool:
        self.cleanup_calls.append(session_id)
        return len(self.cleanup_calls) == 1


@pytest.mark.asyncio
async def test_rag_index_retrieve_and_cleanup_endpoints(monkeypatch):
    fake_agent = FakeRAGAgent()
    monkeypatch.setattr(api_main, "medical_rag_agent", fake_agent)

    doc = MedicalDocumentSchema(
        document_type=DocumentType.PDF,
        raw_text="sample",
        chunks=[{"text": "Patient has chest pain.", "section": "HPI", "metadata": {}}],
        metadata={"filename": "patient.pdf"},
    )

    index_payload = RAGIndexPatientRequest(session_id="session-api", documents=[doc])
    index_response = await api_main.rag_index_patient(index_payload)
    assert index_response.session_id == "session-api"
    assert index_response.documents_indexed == 1
    assert fake_agent.index_calls == [("session-api", 1)]

    retrieve_payload = RAGRetrieveRequest(
        query="Chest pain guidance",
        session_id="session-api",
        top_k_patient=3,
        top_k_global=3,
        top_k_total=3,
    )
    retrieve_response = await api_main.rag_retrieve(retrieve_payload)
    assert retrieve_response.query == "Chest pain guidance"
    assert len(retrieve_response.results) == 1
    assert retrieve_response.results[0].origin == "patient_store"

    cleanup_1 = await api_main.rag_cleanup_session("session-api")
    cleanup_2 = await api_main.rag_cleanup_session("session-api")
    assert cleanup_1.deleted is True
    assert cleanup_2.deleted is False
