"""
src/rag/embedding_service.py
----------------------------
Handles the generation of dense vector embeddings for text chunks.
Uses SentenceTransformers to encode medical text into uniform semantic vectors.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import path on limited envs
    SentenceTransformer = None  # type: ignore[assignment]

from src.utils.logger import get_logger
from src.services.redis_cache_service import RedisCacheService

logger = get_logger(__name__)


class _HashingFallbackEmbedder:
    """
    Deterministic offline embedder fallback using hashing features.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(1, 2),
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray()


class _NVIDIAAPIEmbedder:
    """
    Embedding backend powered by NVIDIA Integrate API.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        truncate: str = "NONE",
        timeout_seconds: float = 60.0,
        max_batch_size: int = 32,
    ) -> None:
        if not api_key:
            raise RuntimeError("NVIDIA API key is missing.")
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.truncate = truncate
        self.max_batch_size = max(1, max_batch_size)
        self._timeout_seconds = timeout_seconds
        self._client = httpx.Client(timeout=self._timeout_seconds)

    def _request_embeddings(self, inputs: Sequence[str]) -> np.ndarray:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "input": list(inputs),
            "encoding_format": "float",
            "truncate": self.truncate,
        }
        response = self._client.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        raw_data = body.get("data", [])
        if not raw_data:
            raise RuntimeError("NVIDIA embeddings API returned an empty `data` payload.")

        data_sorted = sorted(raw_data, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in data_sorted]
        if not vectors or not isinstance(vectors[0], list):
            raise RuntimeError("NVIDIA embeddings API returned invalid vector payload format.")
        return np.asarray(vectors, dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        batches = []
        for start_idx in range(0, len(texts), self.max_batch_size):
            batch = texts[start_idx:start_idx + self.max_batch_size]
            batches.append(self._request_embeddings(batch))

        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        provider: str = "nvidia_api",
        fallback_dimension: int = 384,
        local_files_only: bool = True,
        nvidia_api_url: str = "https://integrate.api.nvidia.com/v1/embeddings",
        nvidia_api_key: str = "",
        nvidia_truncate: str = "NONE",
        request_timeout_seconds: float = 60.0,
        nvidia_max_batch_size: int = 32,
        cache_service: Optional[RedisCacheService] = None,
        cache_ttl_seconds: int = 7 * 24 * 60 * 60,
    ):
        """
        Initializes the embedding service with a provider backend.
        Supported providers:
          - nvidia_api
          - sentence_transformer
        """
        self.backend_name = ""
        self.model_name = model_name
        self.provider = provider
        self.cache_service = cache_service
        self.cache_ttl_seconds = cache_ttl_seconds

        try:
            if self.provider == "nvidia_api":
                logger.info(
                    "loading_embedding_model",
                    provider=self.provider,
                    model_name=model_name,
                    api_url=nvidia_api_url,
                )
                self.model = _NVIDIAAPIEmbedder(
                    api_url=nvidia_api_url,
                    api_key=nvidia_api_key,
                    model_name=model_name,
                    truncate=nvidia_truncate,
                    timeout_seconds=request_timeout_seconds,
                    max_batch_size=nvidia_max_batch_size,
                )
                probe = self.model.encode(["embedding_dimension_probe"])
                self.dimension = int(probe.shape[1])
                self.backend_name = "nvidia_api"
            elif self.provider == "sentence_transformer":
                if SentenceTransformer is None:
                    raise RuntimeError("sentence-transformers package is unavailable")
                logger.info(
                    "loading_embedding_model",
                    provider=self.provider,
                    model_name=model_name,
                    local_files_only=local_files_only,
                )
                self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.backend_name = "sentence_transformer"
            else:
                raise RuntimeError(f"Unsupported embedding provider: {self.provider}")

            logger.info(
                "embedding_model_loaded",
                backend=self.backend_name,
                provider=self.provider,
                model_name=model_name,
                dimension=self.dimension,
            )
        except Exception as exc:
            logger.warning(
                "embedding_model_load_failed_fallback_enabled",
                provider=self.provider,
                model_name=model_name,
                error=str(exc),
            )
            self.model = _HashingFallbackEmbedder(dimension=fallback_dimension)
            self.dimension = fallback_dimension
            self.backend_name = "hashing_fallback"
            logger.info(
                "embedding_fallback_loaded",
                backend=self.backend_name,
                dimension=self.dimension,
            )

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)

    @staticmethod
    def _normalize_cache_text(text: str) -> str:
        # Normalize only surrounding/repeated whitespace to keep semantics stable.
        return " ".join((text or "").strip().split())

    def _embedding_cache_key(self, text: str) -> str:
        normalized = self._normalize_cache_text(text)
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return (
            f"emb:v1:provider:{self.provider}:backend:{self.backend_name}:"
            f"model:{self.model_name}:dim:{self.dimension}:text:{text_hash}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string into a 1D numpy array.
        """
        embedding = self.embed_batch([text or ""])
        return embedding[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of strings into a 2D numpy array.
        Returns shape: (num_texts, dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        safe_texts = [text if isinstance(text, str) else str(text) for text in texts]

        cache = self.cache_service
        if cache is None:
            if self.backend_name == "sentence_transformer":
                embeddings = self.model.encode(safe_texts, convert_to_numpy=True, show_progress_bar=False)
            else:
                embeddings = self.model.encode(safe_texts)
            vectors = embeddings.astype(np.float32)
            return self._normalize_rows(vectors)

        rows: list[np.ndarray | None] = [None] * len(safe_texts)
        misses: list[str] = []
        miss_indices: list[int] = []
        miss_keys: list[str] = []

        for idx, text in enumerate(safe_texts):
            key = self._embedding_cache_key(text)
            cached = cache.get_json(key)
            if (
                isinstance(cached, list)
                and len(cached) == self.dimension
            ):
                try:
                    rows[idx] = np.asarray(cached, dtype=np.float32)
                    continue
                except Exception:
                    pass
            misses.append(text)
            miss_indices.append(idx)
            miss_keys.append(key)

        if misses:
            if self.backend_name == "sentence_transformer":
                uncached = self.model.encode(misses, convert_to_numpy=True, show_progress_bar=False)
            else:
                uncached = self.model.encode(misses)

            normalized_uncached = self._normalize_rows(uncached.astype(np.float32))
            for offset, row_idx in enumerate(miss_indices):
                vector = normalized_uncached[offset]
                rows[row_idx] = vector
                cache.set_json(
                    miss_keys[offset],
                    vector.tolist(),
                    ttl_seconds=self.cache_ttl_seconds,
                )

        output = np.vstack([
            row if row is not None else np.zeros((self.dimension,), dtype=np.float32)
            for row in rows
        ]).astype(np.float32)
        return output
