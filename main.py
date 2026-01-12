import asyncio
import argparse
import yaml
import logging
import time
import json
import csv
import os
from datetime import datetime as dt
from typing import List

import aiohttp
from colorama import Fore, Style, init

from src.client.api_client import LoadTester
from src.generators.prompt_factory import PromptFactory
from src.metrics.stats import StatsCalculator, RequestMetrics

# Initialize colorama
init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def user_session(
    client: LoadTester,
    prompt_factory: PromptFactory,
    stats: StatsCalculator,
    config: dict,
    end_time: float
):
    """
    Simulates a single user sending requests until end_time is reached.
    """
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            prompt = prompt_factory.generate_prompt()
            
            # Send request
            metric = await client.send_request(session, prompt, config["prompts"])
            
            # Record metrics
            stats.add_metric(metric)
            
            # Optional: Add think time/sleep between requests if needed
            # await asyncio.sleep(0.1) 

async def main():
    parser = argparse.ArgumentParser(description="LLM Load Testing Tool")
    parser.add_argument("--config", default="config/workload.yaml", help="Path to configuration file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    server_conf = config["server"]
    workload_conf = config["workload"]
    prompts_conf = config["prompts"]

    client = LoadTester(server_conf["base_url"], server_conf.get("model_alias", ""))
    prompt_factory = PromptFactory(
        seed=workload_conf["seed"],
        min_tokens=prompts_conf["min_tokens"],
        max_tokens=prompts_conf["max_tokens"],
        prefix=prompts_conf.get("prefix", "")
    )
    stats = StatsCalculator()

    # Create Output Directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

    # Execution Parameters
    total_duration = workload_conf["duration_seconds"]
    concurrency = workload_conf["users"]
    ramp_up = workload_conf.get("ramp_up_seconds", 0)
    
    start_time = time.time()
    end_time = start_time + total_duration
    
    print(f"{Fore.CYAN}Starting load test with {concurrency} users for {total_duration}s...")
    print(f"{Fore.CYAN}Target: {server_conf['base_url']}")

    stats.start_test()
    
    tasks = []
    
    if ramp_up > 0:
        # Ramp-up scenario
        delay_per_user = ramp_up / concurrency
        for i in range(concurrency):
            # Calculate remaining time for this user
            # They start later, but stop at the same global end_time
            task = asyncio.create_task(user_session(client, prompt_factory, stats, config, end_time))
            tasks.append(task)
            if i < concurrency - 1:
                await asyncio.sleep(delay_per_user)
    else:
        # Static concurrency
        for _ in range(concurrency):
            task = asyncio.create_task(user_session(client, prompt_factory, stats, config, end_time))
            tasks.append(task)

    # Wait for completion
    await asyncio.gather(*tasks)
    stats.stop_test()
    
    # Reports
    summary = stats.calculate_summary()
    
    print(f"\n{Fore.GREEN}=== Test Summary ===")
    print(f"Total Requests: {summary.total_requests}")
    print(f"Successful: {summary.successful_requests}")
    print(f"Failed: {summary.failed_requests}")
    print(f"Duration: {summary.total_duration:.2f}s")
    print(f"RPS: {summary.rps:.2f}")
    print(f"Global Throughput: {summary.global_throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"\n{Fore.YELLOW}=== Latency (s) ===")
    print(f"P50: {summary.latency_p50:.3f}")
    print(f"P90: {summary.latency_p90:.3f}")
    print(f"P99: {summary.latency_p99:.3f}")
    print(f"\n{Fore.YELLOW}=== TTFT (s) ===")
    print(f"P50: {summary.ttft_p50:.3f}")
    print(f"P90: {summary.ttft_p90:.3f}")
    print(f"P99: {summary.ttft_p99:.3f}")

    # Save details
    detailed_metrics = [
        {
            "request_id": m.request_id,
            "start_ts": m.start_time,
            "end_ts": m.end_time,
            "ttft": m.ttft,
            "latency": m.latency_seconds,
            "output_tokens": m.output_tokens,
            "tps": m.tokens_per_second,
            "error": m.error
        }
        for m in stats.metrics
    ]
    
    json_path = os.path.join(args.output, f"test_run_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": config,
            "summary": str(summary), # Simple string dump for summary
            "details": detailed_metrics
        }, f, indent=2)
    
    print(f"\n{Fore.BLUE}Detailed logs saved to: {json_path}")

if __name__ == "__main__":
    asyncio.run(main())
