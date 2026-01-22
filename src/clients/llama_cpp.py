import aiohttp
import asyncio
import time
import json
import logging
import uuid
from typing import Dict, Any, Optional

from src.clients.base import BaseLLMClient
from src.metrics.stats import RequestMetrics
from src.config.schema import ServerConfig, ClientConfig

logger = logging.getLogger(__name__)

class LlamaCppClient(BaseLLMClient):
    def __init__(self, server_conf: ServerConfig):
        self.server_conf = server_conf

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        client_config: ClientConfig,
        input_tokens: int = 0
    ) -> RequestMetrics:
        req_id = str(uuid.uuid4())
        start_time = time.time()
        ttft = 0.0
        output_tokens = 0
        error = None
        
        base_url = self.server_conf.base_url
        endpoint_name = self.server_conf.name
        
        # llama.cpp specific payload
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": max_tokens,
        }

        retries = client_config.retries
        backoff_factor = client_config.backoff_factor

        for attempt in range(retries + 1):
            try:
                ttft = 0.0 # Reset for retry
                output_tokens = 0
                error = None
                
                async with session.post(base_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error = f"HTTP {response.status}: {error_text}"
                        if response.status >= 500:
                            if attempt < retries:
                                logger.warning(f"Request {req_id} attempt {attempt+1} failed ({error}). Retrying...")
                                await asyncio.sleep(backoff_factor ** attempt)
                                continue
                        break

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
                    
                    # Success
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
            input_tokens=input_tokens,
            error=error,
            endpoint=endpoint_name
        )
