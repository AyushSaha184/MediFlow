"""
src/services/llm_service.py
---------------------------
Handles communication with external LLM APIs (OpenRouter acting as unified gateway).
Provides abstractions for JSON-structured generation with Temperature control for passes.
"""

from cerebras.cloud.sdk import Cerebras
from typing import Dict, Any, Optional
from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or settings.cerebras_api_key
        self.model_name = settings.diagnostic_model_name
        self._client = None
        if self.api_key:
            self._client = Cerebras(api_key=self.api_key)

    async def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """
        Sends a request to the LLM via Cerebras SDK.
        """
        if not self._client:
            logger.warning("missing_cerebras_api_key", msg="CEREBRAS_API_KEY is not set.")
            raise ValueError("Cerebras API Key is missing.")

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
            logger.info("llm_request_success")
            return content
        except Exception as e:
            logger.error("llm_internal_error", error=str(e), provider="Cerebras")
            raise
