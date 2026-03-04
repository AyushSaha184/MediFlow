"""
Phase 4 tests for MedicalRAGAgent dual-store behavior.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from src.agents.medical_rag_agent import MedicalRAGAgent
from src.models.medical_document import DocumentType, MedicalDocumentSchema
from src.rag.faiss_store import FAISSStore


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.dimension = 8
        self.model_name = "fake-local"
        self.backend_name = "fake-local"

    def _vectorize(self, text: str) -> np.ndarray:
        text_l = text.lower()
        vector = np.zeros(self.dimension, dtype=np.float32)
        if any(token in text_l for token in ("chest", "myocardial", "heart")):
            vector[0] = 2.0
        if any(token in text_l for token in ("arm", "numb")):
            vector[1] = 1.5
        if any(token in text_l for token in ("diabetes", "insulin", "metformin", "glucose")):
            vector[2] = 2.0
        if any(token in text_l for token in ("treatment", "therapy", "plan")):
            vector[3] = 1.0
        if not vector.any():
            vector[4] = 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self._vectorize(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = np.vstack([self._vectorize(text) for text in texts]).astype(np.float32)
        return vectors


@pytest.fixture
def sandbox_dir() -> Path:
    base_dir = Path("data") / "test_rag_agent"
    base_dir.mkdir(parents=True, exist_ok=True)
    created = base_dir / f"rag_agent_{uuid4().hex}"
    created.mkdir(parents=True, exist_ok=True)
    try:
        yield created
    finally:
        shutil.rmtree(created, ignore_errors=True)


def _create_global_store(global_store_dir: Path, embedder: FakeEmbeddingService) -> None:
    store = FAISSStore(dimension=embedder.dimension)
    record = {
        "chunk_id": "global-diabetes",
        "document_id": "global::sample_diabetes.md",
        "source_file": "sample_diabetes.md",
        "origin": "global_store",
        "section": "TREATMENT",
        "text": "Diabetes treatment includes insulin and metformin.",
        "metadata": {"source_type": "guideline"},
    }
    store.add(embedder.embed_batch([record["text"]]), [record])
    store.save_local(str(global_store_dir))


@pytest.mark.asyncio
async def test_medical_rag_patient_ingestion_and_retrieval(sandbox_dir: Path):
    embedder = FakeEmbeddingService()
    global_store_dir = sandbox_dir / "global_store"
    patient_root = sandbox_dir / "patient_root"
    _create_global_store(global_store_dir, embedder)

    rag_agent = MedicalRAGAgent(
        embedder=embedder,
        global_store_dir=global_store_dir,
        patient_data_root=patient_root,
    )

    doc = MedicalDocumentSchema(
        document_type=DocumentType.PDF,
        raw_text="Test Data",
        chunks=[
            {"text": "Patient has severe chest pain.", "section": "HPI", "metadata": {}},
            {"text": "Patient arm feels numb.", "section": "GENERAL", "metadata": {}},
        ],
        metadata={"filename": "test_patient.pdf"},
        processed_by=["DataPrepAgent"],
    )

    stats = await rag_agent.ingest_patient_documents("session-a", [doc])

    assert stats["chunks_seen"] == 2
    assert stats["chunks_embedded"] == 2
    assert "MedicalRAGAgent" in doc.processed_by

    store_path = patient_root / "session-a" / "rag_patient"
    assert (store_path / "index.faiss").exists()
    assert (store_path / "metadata.json").exists()

    results = rag_agent.retrieve(
        query="Possible myocardial infarction symptoms",
        session_id="session-a",
        top_k_patient=5,
        top_k_global=5,
        top_k_total=5,
    )

    assert len(results) > 0
    best_match = results[0]
    assert best_match["source_file"] == "test_patient.pdf"
    assert best_match["origin"] == "patient_store"
    assert "chest pain" in best_match["text"].lower()
    assert "distance" in best_match
    assert "l2_distance" in best_match


def test_medical_rag_global_retrieval_without_session(sandbox_dir: Path):
    embedder = FakeEmbeddingService()
    global_store_dir = sandbox_dir / "global_store"
    _create_global_store(global_store_dir, embedder)

    rag_agent = MedicalRAGAgent(
        embedder=embedder,
        global_store_dir=global_store_dir,
        patient_data_root=sandbox_dir / "patient_root",
    )

    results = rag_agent.retrieve(
        query="What is standard treatment for diabetes?",
        session_id=None,
        top_k_patient=3,
        top_k_global=5,
        top_k_total=3,
    )

    assert len(results) >= 1
    assert results[0]["origin"] == "global_store"
    assert results[0]["source_file"] == "sample_diabetes.md"


@pytest.mark.asyncio
async def test_medical_rag_cleanup_session_scoped_only(sandbox_dir: Path):
    embedder = FakeEmbeddingService()
    rag_agent = MedicalRAGAgent(
        embedder=embedder,
        global_store_dir=sandbox_dir / "global_store",
        patient_data_root=sandbox_dir / "patient_root",
    )

    doc = MedicalDocumentSchema(
        document_type=DocumentType.PDF,
        raw_text="Test Data",
        chunks=[{"text": "Chest pain", "section": "HPI", "metadata": {}}],
        metadata={"filename": "test_cleanup.pdf"},
    )

    await rag_agent.ingest_patient_documents("session-cleanup", [doc])
    session_dir = sandbox_dir / "patient_root" / "session-cleanup" / "rag_patient"
    assert session_dir.exists()

    assert rag_agent.cleanup_session("session-cleanup") is True
    assert session_dir.exists() is False
    assert rag_agent.cleanup_session("session-cleanup") is False
