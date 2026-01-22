import random
import aiohttp
from typing import List, Tuple
from src.clients.base import BaseLLMClient
from src.config.schema import ClientConfig, ServerConfig
from src.metrics.stats import RequestMetrics

class WeightedCompositeClient(BaseLLMClient):
    def __init__(self, clients: List[Tuple[BaseLLMClient, float]]):
        """
        Args:
            clients: List of (client, weight) tuples.
        """
        self.clients = clients
        
        # Normalize weights
        total_weight = sum(w for _, w in clients)
        self.normalized_clients = [
            (client, w / total_weight) for client, w in clients
        ]

    def _select_client(self) -> BaseLLMClient:
        rand = random.random()
        cumulative = 0.0
        for client, weight in self.normalized_clients:
            cumulative += weight
            if rand <= cumulative:
                return client
        return self.normalized_clients[-1][0]

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        client_config: ClientConfig,
        input_tokens: int = 0
    ) -> RequestMetrics:
        client = self._select_client()
        return await client.send_request(session, prompt, max_tokens, client_config, input_tokens=input_tokens)
