"""
Tests for Phase 4 embedding service behavior.
"""

from __future__ import annotations

import numpy as np

from src.rag import embedding_service as embedding_module
from src.rag.embedding_service import EmbeddingService


def test_embedding_service_uses_fallback_when_transformer_unavailable(monkeypatch):
    monkeypatch.setattr(embedding_module, "SentenceTransformer", None)

    service = EmbeddingService(model_name="non-existent-model", fallback_dimension=16)
    vectors = service.embed_batch(["alpha", "beta"])

    assert service.backend_name == "hashing_fallback"
    assert service.dimension == 16
    assert vectors.shape == (2, 16)
    assert vectors.dtype == np.float32


def test_embedding_service_fallback_embeddings_are_deterministic(monkeypatch):
    monkeypatch.setattr(embedding_module, "SentenceTransformer", None)

    service = EmbeddingService(model_name="non-existent-model", fallback_dimension=32)
    vec_1 = service.embed_text("patient has chest pain")
    vec_2 = service.embed_text("patient has chest pain")
    vec_3 = service.embed_text("patient has diabetes")

    assert vec_1.shape[0] == 32
    assert np.allclose(vec_1, vec_2)
    assert not np.allclose(vec_1, vec_3)
