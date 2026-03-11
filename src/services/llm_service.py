"""
src/services/llm_service.py
---------------------------
Handles communication with external LLM APIs (OpenRouter acting as unified gateway).
Provides abstractions for JSON-structured generation with Temperature control for passes.
"""

import hashlib
from cerebras.cloud.sdk import Cerebras
from typing import Optional
from src.core.config import settings
from src.utils.logger import get_logger
from src.services.redis_cache_service import RedisCacheService

logger = get_logger(__name__)

class LLMService:
    def __init__(
        self,
        api_key: str = "",
        cache_service: Optional[RedisCacheService] = None,
        cache_ttl_seconds: int = 1800,
    ):
        self.api_key = api_key or settings.cerebras_api_key
        self.model_name = settings.diagnostic_model_name
        self.cache_service = cache_service
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = None
        if self.api_key:
            self._client = Cerebras(api_key=self.api_key)

    def _cache_key(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        payload = f"{self.model_name}\n{temperature}\n{system_prompt}\n---\n{user_prompt}"
        prompt_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"llm:v1:model:{self.model_name}:temp:{temperature:.3f}:prompt:{prompt_hash}"

    async def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """
        Sends a request to the LLM via Cerebras SDK.
        """
        if not self._client:
            logger.warning("missing_cerebras_api_key", msg="CEREBRAS_API_KEY is not set.")
            raise ValueError("Cerebras API Key is missing.")

        is_deterministic = abs(float(temperature)) < 1e-9
        cache_key = self._cache_key(system_prompt, user_prompt, temperature)
        if is_deterministic and self.cache_service is not None:
            cached = self.cache_service.get_json(cache_key)
            if isinstance(cached, str) and cached:
                logger.info("llm_request_cache_hit", model=self.model_name, temp=temperature)
                return cached

        logger.info("llm_request_started", model=self.model_name, temp=temperature, provider="Cerebras")
        
        try:
            # Cerebras SDK currently uses synchronous calls in their main completions
            # We wrap it in a thread pool for async compatibility if needed, 
            # but usually the SDK might have an async variant or we use the sync one directly here.
            # Assuming standard sync SDK for now based on user's snippet.
            response = self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=temperature,
                # Cerebras typically follows OpenAI format
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if is_deterministic and self.cache_service is not None and content:
                self.cache_service.set_json(
                    cache_key,
                    content,
                    ttl_seconds=self.cache_ttl_seconds,
                )
            logger.info("llm_request_success")
            return content
        except Exception as e:
            logger.error("llm_internal_error", error=str(e), provider="Cerebras")
            raise
