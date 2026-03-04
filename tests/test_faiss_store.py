"""
Tests for FAISSStore wrappers used by Phase 4.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.rag.faiss_store import FAISSStore


def test_faiss_store_add_search_and_filter():
    store = FAISSStore(dimension=4, required_metadata_keys={"chunk_id", "origin"})
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadatas = [
        {"chunk_id": "chunk-a", "origin": "patient_store", "text": "chest pain"},
        {"chunk_id": "chunk-b", "origin": "global_store", "text": "diabetes treatment"},
    ]

    added = store.add(embeddings=embeddings, metadatas=metadatas)
    assert added == 2

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, k=2)
    assert len(results) == 2
    assert results[0][1]["chunk_id"] == "chunk-a"

    filtered = store.search(query, k=2, metadata_filter={"origin": "global_store"})
    assert len(filtered) == 1
    assert filtered[0][1]["chunk_id"] == "chunk-b"


def test_faiss_store_dedupes_by_chunk_id():
    store = FAISSStore(dimension=4)
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    metadata = [{"chunk_id": "chunk-same", "origin": "patient_store"}]

    assert store.add(embeddings=embeddings, metadatas=metadata, dedupe_by_chunk_id=True) == 1
    assert store.add(embeddings=embeddings, metadatas=metadata, dedupe_by_chunk_id=True) == 0
    assert store.index.ntotal == 1


def test_faiss_store_save_load_roundtrip():
    root = Path("data") / "test_faiss_store"
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f"faiss_roundtrip_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        store = FAISSStore(dimension=4)
        embeddings = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        metadata = [{"chunk_id": "chunk-roundtrip", "origin": "global_store"}]
        store.add(embeddings=embeddings, metadatas=metadata)
        store.save_local(str(tmp_dir))

        loaded = FAISSStore.load_local(str(tmp_dir), dimension=4)
        assert loaded.index.ntotal == 1
        assert len(loaded.metadata_store) == 1
        assert loaded.metadata_store[0]["chunk_id"] == "chunk-roundtrip"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
