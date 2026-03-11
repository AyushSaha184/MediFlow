"""
src/services/redis_cache_service.py
-----------------------------------
Best-effort Redis cache wrapper used for embeddings, retrieval, and LLM outputs.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]


class RedisCacheService:
    def __init__(
        self,
        redis_url: str = "",
        key_prefix: str = "mediflow",
        enabled: bool = True,
        socket_timeout_seconds: float = 1.0,
    ) -> None:
        self.redis_url = (redis_url or "").strip()
        self.key_prefix = key_prefix
        self.enabled = enabled and bool(self.redis_url) and redis is not None
        self._socket_timeout_seconds = socket_timeout_seconds
        self._client: Optional["redis.Redis"] = None
        self._disabled_reason = ""

        if redis is None:
            self.enabled = False
            self._disabled_reason = "redis package unavailable"
        elif not self.redis_url:
            self.enabled = False
            self._disabled_reason = "REDIS_URL unset"

        if not self.enabled:
            logger.info("redis_cache_disabled", reason=self._disabled_reason or "disabled")

    def _ensure_client(self) -> Optional["redis.Redis"]:
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = redis.Redis.from_url(  # type: ignore[union-attr]
                self.redis_url,
                decode_responses=True,
                socket_timeout=self._socket_timeout_seconds,
                socket_connect_timeout=self._socket_timeout_seconds,
            )
            self._client.ping()
            logger.info("redis_cache_connected", redis_url=self.redis_url)
            return self._client
        except Exception as exc:
            self.enabled = False
            self._disabled_reason = str(exc)
            logger.warning("redis_cache_init_failed", error=str(exc))
            return None

    def _k(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            raw = client.get(self._k(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("redis_cache_get_failed", key=key, error=str(exc))
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        client = self._ensure_client()
        if client is None:
            return
        try:
            payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            client.setex(self._k(key), ttl_seconds, payload)
        except Exception as exc:
            logger.warning("redis_cache_set_failed", key=key, error=str(exc))
