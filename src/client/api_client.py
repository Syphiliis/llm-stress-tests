import aiohttp
import asyncio
import time
import json
import logging
import uuid
import random
from typing import Optional, Dict, Any, List

from src.metrics.stats import RequestMetrics

logger = logging.getLogger(__name__)

class LoadTester:
    def __init__(self, base_url: str = None, model_alias: str = None, servers: List[Dict] = None):
        """
        Initialize LoadTester with either a single endpoint or multiple endpoints.

        Args:
            base_url: Single endpoint URL (for simple scenarios)
            model_alias: Single model alias (for simple scenarios)
            servers: List of server configurations for mixed warfare scenario
                     Each server dict should have: name, base_url, model_alias, weight
        """
        if servers:
            # Mixed warfare mode
            self.servers = servers
            self.mixed_mode = True
            # Normalize weights
            total_weight = sum(s.get('weight', 1.0) for s in servers)
            for server in self.servers:
                server['normalized_weight'] = server.get('weight', 1.0) / total_weight
        else:
            # Single endpoint mode
            self.base_url = base_url
            self.model_alias = model_alias
            self.mixed_mode = False
            self.servers = None

    def select_server(self) -> Dict[str, Any]:
        """
        Select a server based on configured weights (for mixed warfare).

        Returns:
            Server configuration dict
        """
        if not self.mixed_mode:
            return {
                'name': self.model_alias or 'default',
                'base_url': self.base_url,
                'model_alias': self.model_alias
            }

        # Weighted random selection
        rand = random.random()
        cumulative = 0.0
        for server in self.servers:
            cumulative += server['normalized_weight']
            if rand <= cumulative:
                return server

        # Fallback to last server
        return self.servers[-1]

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        config: Dict[str, Any],
        client_config: Dict[str, Any]
    ) -> RequestMetrics:
        req_id = str(uuid.uuid4())
        start_time = time.time()
        ttft = 0.0
        output_tokens = 0
        error = None
        
        # Retry configuration
        retries = client_config.get("retries", 3)
        backoff_factor = client_config.get("backoff_factor", 1.5)

        # Select server (for mixed warfare or single endpoint)
        server = self.select_server()
        base_url = server['base_url']
        endpoint_name = server.get('name', 'default')

        # llama.cpp specific payload
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": config.get("max_tokens", 200),
        }

        for attempt in range(retries + 1):
            try:
                ttft = 0.0 # Reset for retry
                output_tokens = 0
                error = None
                
                async with session.post(base_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error = f"HTTP {response.status}: {error_text}"
                        # Retry on server errors (5xx)
                        if response.status >= 500:
                            if attempt < retries:
                                logger.warning(f"Request {req_id} attempt {attempt+1} failed ({error}). Retrying...")
                                await asyncio.sleep(backoff_factor ** attempt)
                                continue
                        break # Stop checking attempts for non-retriable errors

                    # Process streaming response
                    async for line in response.content:
                        line = line.strip()
                        if not line or line == b": ping - keepalive":
                            continue

                        if line.startswith(b"data: "):
                            data_str = line.decode("utf-8")[6:]
                            
                            try:
                                data = json.loads(data_str)
                                
                                if output_tokens == 0:
                                    ttft = time.time() - start_time
                                
                                content = data.get("content", "")
                                if content:
                                    output_tokens += 1
                                    
                                if data.get("stop", False):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    # If we reached here without exception, success
                    break

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = str(e)
                if attempt < retries:
                    logger.warning(f"Request {req_id} attempt {attempt+1} failed with connection error: {e}. Retrying...")
                    await asyncio.sleep(backoff_factor ** attempt)
                else:
                    logger.error(f"Request {req_id} failed after {retries} retries: {e}")
            except Exception as e:
                error = f"Unexpected error: {e}"
                logger.error(f"Request {req_id} failed with unexpected error: {e}")
                break

        end_time = time.time()
        
        if ttft == 0.0: 
             ttft = end_time - start_time

        if not error:
             logger.info(f"Request {req_id} [{endpoint_name}] finished: {output_tokens} tokens in {end_time - start_time:.2f}s")
        
        return RequestMetrics(
            request_id=req_id,
            start_time=start_time,
            end_time=end_time,
            ttft=ttft,
            output_tokens=output_tokens,
            error=error,
            endpoint=endpoint_name
        )
