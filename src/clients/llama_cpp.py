import aiohttp
import asyncio
import time
import json
import logging
import uuid
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from src.clients.base import BaseLLMClient
from src.metrics.stats import RequestMetrics
from src.config.schema import ServerConfig, ClientConfig

logger = logging.getLogger(__name__)

class LlamaCppClient(BaseLLMClient):
    def __init__(self, server_conf: ServerConfig):
        self.server_conf = server_conf
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        path = urlparse(self.server_conf.base_url).path.rstrip("/")
        if path.endswith("/api/generate"):
            return "ollama"
        return "llama_cpp"

    def _build_payload(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        if self.backend == "ollama":
            if not self.server_conf.model_alias:
                raise ValueError(
                    "Ollama /api/generate requires 'model_alias' so the request can set the model name."
                )
            return {
                "model": self.server_conf.model_alias,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                },
            }

        return {
            "prompt": prompt,
            "stream": True,
            "n_predict": max_tokens,
        }

    def _decode_stream_chunk(self, raw_line: bytes) -> Optional[Dict[str, Any]]:
        line = raw_line.strip()
        if not line or line == b": ping - keepalive":
            return None

        if line == b"data: [DONE]":
            return {"stop": True}

        if line.startswith(b"data: "):
            line = line[6:]

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _extract_chunk_text(self, data: Dict[str, Any]) -> str:
        if self.backend == "ollama":
            return data.get("response", "")
        return data.get("content", "")

    def _is_done_chunk(self, data: Dict[str, Any]) -> bool:
        if self.backend == "ollama":
            return bool(data.get("done", False))
        return bool(data.get("stop", False))

    def _update_output_token_count(self, current_count: int, data: Dict[str, Any]) -> int:
        if self.backend == "ollama":
            eval_count = data.get("eval_count")
            if isinstance(eval_count, int) and eval_count >= 0:
                return eval_count

        return current_count + 1 if self._extract_chunk_text(data) else current_count

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
        payload = self._build_payload(prompt, max_tokens)

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
                        data = self._decode_stream_chunk(line)
                        if not data:
                            continue

                        content = self._extract_chunk_text(data)
                        if content and output_tokens == 0:
                            ttft = time.time() - start_time

                        output_tokens = self._update_output_token_count(output_tokens, data)

                        if self._is_done_chunk(data):
                            break
                    
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
