import aiohttp
import time
import json
import logging
import uuid
from typing import Optional, Dict, Any

from src.metrics.stats import RequestMetrics

logger = logging.getLogger(__name__)

class LoadTester:
    def __init__(self, base_url: str, model_alias: str):
        self.base_url = base_url
        self.model_alias = model_alias

    async def send_request(
        self, 
        session: aiohttp.ClientSession, 
        prompt: str, 
        config: Dict[str, Any]
    ) -> RequestMetrics:
        req_id = str(uuid.uuid4())
        start_time = time.time()
        ttft = 0.0
        output_tokens = 0
        error = None
        
        # llama.cpp specific payload
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": config.get("max_tokens", 200), # Ensure we limit generation
            # Add other sampling params if needed from config
        }

        try:
            async with session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    error = f"HTTP {response.status}: {await response.text()}"
                    return RequestMetrics(
                        request_id=req_id,
                        start_time=start_time,
                        end_time=time.time(),
                        error=error
                    )

                # Process streaming response
                async for line in response.content:
                    line = line.strip()
                    if not line or line == b": ping - keepalive":
                        continue

                    if line.startswith(b"data: "):
                        data_str = line.decode("utf-8")[6:] # Skip "data: "
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Check if it's the first token
                            if output_tokens == 0:
                                ttft = time.time() - start_time
                            
                            content = data.get("content", "")
                            if content:
                                output_tokens += 1
                                
                            if data.get("stop", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            error = str(e)
            logger.error(f"Request {req_id} failed: {e}")

        end_time = time.time()
        
        # If we didn't receive any tokens (e.g. error before first token), ttft is effectively the full duration
        # but technically it's undefined. We'll set it to duration for stat purposes or 0.
        if ttft == 0.0: 
             ttft = end_time - start_time

        logger.info(f"Request {req_id} finished: {output_tokens} tokens in {end_time - start_time:.2f}s")
        return RequestMetrics(
            request_id=req_id,
            start_time=start_time,
            end_time=end_time,
            ttft=ttft,
            output_tokens=output_tokens,
            error=error
        )
