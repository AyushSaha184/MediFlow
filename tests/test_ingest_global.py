"""
Tests for global knowledge ingestion append mode.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.rag.faiss_store import FAISSStore
from src.rag import ingest_global as ingest_module


class FakeEmbeddingService:
    def __init__(self, model_name: str = "fake", **_: object) -> None:
        self.model_name = model_name
        self.dimension = 12
        self.backend_name = "fake"

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for idx, text in enumerate(texts):
            slot = abs(hash(text)) % self.dimension
            matrix[idx, slot] = 1.0
        return matrix


def test_ingest_global_append_mode(monkeypatch):
    sandbox_root = Path("data") / "test_ingest_global"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    sandbox = sandbox_root / f"ingest_global_{uuid4().hex}"
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        kb_dir = sandbox / "knowledge_base"
        store_dir = sandbox / "global_store"
        kb_dir.mkdir(parents=True, exist_ok=True)

        (kb_dir / "sample_1.md").write_text(
            "# Diabetes\n\nTREATMENT:\nInsulin and metformin.\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_DIR", str(kb_dir))
        monkeypatch.setattr(ingest_module, "GLOBAL_STORE_DIR", str(store_dir))
        monkeypatch.setattr(ingest_module, "EmbeddingService", FakeEmbeddingService)

        first = ingest_module.run_ingestion()
        assert first["files_scanned"] == 1
        assert first["chunks_added"] > 0

        second = ingest_module.run_ingestion()
        assert second["files_scanned"] == 1
        assert second["chunks_added"] == 0
        assert second["chunks_skipped"] >= first["chunks_added"]

        (kb_dir / "sample_2.txt").write_text("CHEST PAIN treatment includes ECG.\n", encoding="utf-8")
        third = ingest_module.run_ingestion()
        assert third["files_scanned"] == 2
        assert third["chunks_added"] > 0

        store = FAISSStore.load_local(str(store_dir), dimension=12)
        assert store.index.ntotal >= first["chunks_added"] + third["chunks_added"]
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
