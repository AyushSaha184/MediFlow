"""
src/rag/faiss_store.py
----------------------
A wrapper around the FAISS vector database.
Maintains a mapping between FAISS matrix rows (IDs) and JSON metadata.
Provides easy save/load functionality to isolate 'Global' vs 'Patient' namespaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from src.rag.common import flatten_record
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:
    def __init__(self, dimension: int, required_metadata_keys: Optional[Set[str]] = None):
        """
        Initializes an empty FAISS index (L2 distance).
        `dimension` must match the output shape of the EmbeddingService.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []
        self.required_metadata_keys = required_metadata_keys or set()

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("Each metadata entry must be a dictionary.")

        if not self.required_metadata_keys:
            return

        missing = [key for key in self.required_metadata_keys if key not in metadata]
        if missing:
            raise ValueError(f"Metadata is missing required keys: {missing}")

    @staticmethod
    def _extract_chunk_id(metadata: Dict[str, Any]) -> Optional[str]:
        normalized = flatten_record(metadata)
        chunk_id = normalized.get("chunk_id")
        return str(chunk_id) if chunk_id else None

    def existing_chunk_ids(self) -> Set[str]:
        chunk_ids: Set[str] = set()
        for metadata in self.metadata_store:
            chunk_id = self._extract_chunk_id(metadata)
            if chunk_id:
                chunk_ids.add(chunk_id)
        return chunk_ids

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        dedupe_by_chunk_id: bool = False,
    ) -> int:
        """
        Add a batch of dense vectors and their corresponding metadata.
        embeddings shape: (n, dimension)
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, received {embeddings.shape[1]}."
            )

        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Size mismatch between embeddings and metadatas.")

        if embeddings.shape[0] == 0:
            return 0

        for metadata in metadatas:
            self._validate_metadata(metadata)

        selected_vectors: List[np.ndarray] = []
        selected_metadata: List[Dict[str, Any]] = []

        existing_ids = self.existing_chunk_ids() if dedupe_by_chunk_id else set()
        skipped = 0

        for index, metadata in enumerate(metadatas):
            if dedupe_by_chunk_id:
                chunk_id = self._extract_chunk_id(metadata)
                if chunk_id and chunk_id in existing_ids:
                    skipped += 1
                    continue
                if chunk_id:
                    existing_ids.add(chunk_id)

            selected_vectors.append(embeddings[index])
            selected_metadata.append(metadata)

        if not selected_vectors:
            logger.debug("faiss_store_add_skipped_all_duplicates", duplicates=skipped)
            return 0

        vectors = np.vstack(selected_vectors).astype(np.float32)
        self.index.add(vectors)
        self.metadata_store.extend(selected_metadata)

        logger.debug(
            "faiss_store_add",
            added=len(selected_metadata),
            duplicates_skipped=skipped,
            total=self.index.ntotal,
        )
        return len(selected_metadata)

    @staticmethod
    def _metadata_matches_filter(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        normalized = flatten_record(metadata)
        for key, value in metadata_filter.items():
            if normalized.get(key) != value:
                return False
        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the FAISS store for the nearest K matches to a query vector.
        query_vector shape must be (1, dimension).
        Returns tuples of (`l2_distance`, `metadata_dict`). Lower distance is better.
        """
        if k <= 0 or self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            q_vec = np.expand_dims(query_vector, axis=0).astype(np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        if q_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, received {q_vec.shape[1]}."
            )

        search_k = self.index.ntotal if metadata_filter else min(k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata_store):
                continue
            metadata = self.metadata_store[idx]
            if metadata_filter and not self._metadata_matches_filter(metadata, metadata_filter):
                continue
            results.append((float(dist), metadata))
            if len(results) >= k:
                break

        return results

    def save_local(self, directory: str) -> None:
        """
        Saves the FAISS index and the JSON metadata map to the specified directory.
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        index_path = directory_path / "index.faiss"
        meta_path = directory_path / "metadata.json"

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(self.metadata_store, handle, indent=2)

        logger.info("faiss_store_saved", path=str(directory_path), total_documents=self.index.ntotal)

    @staticmethod
    def _truncate_index(index: faiss.IndexFlatL2, target_total: int, dimension: int) -> faiss.IndexFlatL2:
        clipped = faiss.IndexFlatL2(dimension)
        if target_total <= 0:
            return clipped

        vectors = np.vstack([index.reconstruct(i) for i in range(target_total)]).astype(np.float32)
        clipped.add(vectors)
        return clipped

    @classmethod
    def load_local(cls, directory: str, dimension: int = 384, required_metadata_keys: Optional[Set[str]] = None) -> "FAISSStore":
        """
        Loads the FAISS index and metadata if they exist. Otherwise returns an empty store.
        """
        store = cls(dimension=dimension, required_metadata_keys=required_metadata_keys)

        directory_path = Path(directory)
        index_path = directory_path / "index.faiss"
        meta_path = directory_path / "metadata.json"

        if not (index_path.exists() and meta_path.exists()):
            logger.warning("faiss_store_not_found", path=str(directory_path), msg="Starting empty.")
            return store

        store.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                store.metadata_store = payload
            else:
                logger.warning("faiss_store_invalid_metadata_payload", path=str(meta_path))
                store.metadata_store = []

        if store.index.d != store.dimension:
            logger.warning(
                "faiss_store_dimension_override",
                requested_dimension=store.dimension,
                index_dimension=store.index.d,
            )
            store.dimension = store.index.d

        if len(store.metadata_store) != store.index.ntotal:
            target_total = min(len(store.metadata_store), store.index.ntotal)
            logger.warning(
                "faiss_store_alignment_fix",
                index_total=store.index.ntotal,
                metadata_total=len(store.metadata_store),
                aligned_total=target_total,
            )
            store.metadata_store = store.metadata_store[:target_total]
            store.index = cls._truncate_index(store.index, target_total=target_total, dimension=store.dimension)

        logger.info("faiss_store_loaded", path=str(directory_path), total_documents=store.index.ntotal)

        return store
