from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import aiohttp
from src.metrics.stats import RequestMetrics
from src.config.schema import ClientConfig

class BaseLLMClient(ABC):
    @abstractmethod
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        client_config: ClientConfig,
        input_tokens: int = 0
    ) -> RequestMetrics:
        """
        Send a request to the LLM backend.
        
        Args:
            session: The aiohttp session to use.
            prompt: The text prompt.
            max_tokens: Max tokens to generate.
            client_config: Client configuration (retries, backoff, etc).
        
        Returns:
            RequestMetrics object with stats.
        """
        pass
