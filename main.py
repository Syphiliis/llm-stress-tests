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
from src.metrics.prometheus_exporter import PrometheusExporter

# Initialize colorama
init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global counter for tracking active users
active_users_count = 0
active_users_lock = asyncio.Lock()

async def progress_logger_task(
    stats: StatsCalculator,
    start_time: float,
    end_time: float,
    interval: float = 10.0
):
    """
    Background task that periodically logs progress during the test.
    """
    global active_users_count

    while time.time() < end_time:
        await asyncio.sleep(interval)

        # Calculate elapsed time and remaining time
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        # Get current metrics snapshot
        snapshot = stats.get_current_snapshot()

        async with active_users_lock:
            current_active = active_users_count

        # Print progress update
        print(f"\n{Fore.BLUE}[Progress @ {elapsed:.0f}s] "
              f"Active Users: {current_active} | "
              f"Requests: {len(stats.metrics)} | "
              f"RPS: {snapshot['rps']:.1f} | "
              f"TPS: {snapshot['tps']:.1f} tok/s | "
              f"P90 Latency: {snapshot['latency_p90']:.2f}s | "
              f"Errors: {snapshot['error_rate']:.1%} | "
              f"Time Left: {remaining:.0f}s")

    logger.info("Progress logger completed")

async def metrics_pusher_task(
    prom_exporter: PrometheusExporter,
    stats: StatsCalculator,
    push_interval: float,
    end_time: float
):
    """
    Background task that periodically pushes metrics to Prometheus Pushgateway.
    """
    global active_users_count

    while time.time() < end_time:
        await asyncio.sleep(push_interval)

        # Get current metrics snapshot
        snapshot = stats.get_current_snapshot()
        deltas = stats.get_delta_counts()

        # Update Prometheus metrics
        async with active_users_lock:
            current_active = active_users_count

        prom_exporter.update_metrics(snapshot, current_active)
        prom_exporter.increment_counters(
            requests_delta=deltas['requests'],
            errors_delta=deltas['errors'],
            tokens_delta=deltas['tokens']
        )

        # Push to gateway
        success = prom_exporter.push_metrics()
        if success:
            logger.debug(f"Metrics pushed: RPS={snapshot['rps']:.2f}, TPS={snapshot['tps']:.2f}, "
                        f"P90_Lat={snapshot['latency_p90']:.3f}s, Active={current_active}")

    logger.info("Metrics pusher task completed")

async def user_session(
    client: LoadTester,
    prompt_factory: PromptFactory,
    stats: StatsCalculator,
    config: dict,
    end_time: float,
    client_config: dict
):
    """
    Simulates a single user sending requests until end_time is reached.
    """
    global active_users_count

    # Track this user as active
    async with active_users_lock:
        active_users_count += 1

    # Configure timeout
    timeout = aiohttp.ClientTimeout(
        total=client_config.get("timeout_seconds", 60),
        connect=client_config.get("connect_timeout", 10)
    )

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time.time() < end_time:
                prompt = prompt_factory.generate_prompt()

                # Send request
                metric = await client.send_request(session, prompt, config["prompts"], client_config)

                # Record metrics
                stats.add_metric(metric)

                # Optional: Add think time/sleep between requests if needed
                # await asyncio.sleep(0.1)
    finally:
        # Decrement when user session ends
        async with active_users_lock:
            active_users_count -= 1 

async def main():
    parser = argparse.ArgumentParser(description="LLM Load Testing Tool")
    parser.add_argument("--config", default="config/workload.yaml", help="Path to configuration file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    workload_conf = config["workload"]
    prompts_conf = config["prompts"]
    client_conf = config.get("client", {})

    # Check for mixed warfare mode (multiple servers)
    if "servers" in config:
        # Mixed Warfare: multiple endpoints
        servers_conf = config["servers"]
        client = LoadTester(servers=servers_conf)
        logger.info(f"Mixed Warfare mode enabled with {len(servers_conf)} endpoints")
        for srv in servers_conf:
            logger.info(f"  - {srv['name']}: {srv['base_url']} (weight: {srv.get('weight', 1.0)})")
    else:
        # Single endpoint mode
        server_conf = config["server"]
        client = LoadTester(server_conf["base_url"], server_conf.get("model_alias", ""))
    prompt_factory = PromptFactory(
        seed=workload_conf["seed"],
        min_tokens=prompts_conf["min_tokens"],
        max_tokens=prompts_conf["max_tokens"],
        prefix=prompts_conf.get("prefix", "")
    )
    stats = StatsCalculator()

    # Initialize Prometheus exporter if enabled
    prom_exporter = None
    prom_conf = config.get("prometheus", {})
    if prom_conf.get("enabled", False):
        prom_exporter = PrometheusExporter(
            pushgateway_url=prom_conf["pushgateway_url"],
            job_name=prom_conf.get("job_name", "llm_load_test"),
            instance_name=prom_conf.get("instance_name", "local"),
            username=prom_conf.get("username"),
            password=prom_conf.get("password")
        )
        logger.info(f"Prometheus integration enabled: {prom_conf['pushgateway_url']}")

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
    if "servers" in config:
        print(f"{Fore.CYAN}Mode: Mixed Warfare ({len(config['servers'])} endpoints)")
    else:
        print(f"{Fore.CYAN}Target: {server_conf['base_url']}")

    stats.start_test()

    tasks = []

    # Start progress logger
    progress_task = asyncio.create_task(
        progress_logger_task(stats, start_time, end_time, interval=10.0)
    )
    tasks.append(progress_task)

    # Start Prometheus metrics pusher if enabled
    if prom_exporter:
        push_interval = prom_conf.get("push_interval_seconds", 5)
        pusher_task = asyncio.create_task(
            metrics_pusher_task(prom_exporter, stats, push_interval, end_time)
        )
        tasks.append(pusher_task)
        logger.info(f"Metrics will be pushed every {push_interval}s")

    if ramp_up > 0:
        # Ramp-up scenario
        delay_per_user = ramp_up / concurrency
        for i in range(concurrency):
            # Calculate remaining time for this user
            # They start later, but stop at the same global end_time
            # Calculate remaining time for this user
            # They start later, but stop at the same global end_time
            task = asyncio.create_task(user_session(client, prompt_factory, stats, config, end_time, client_conf))
            tasks.append(task)
            if i < concurrency - 1:
                await asyncio.sleep(delay_per_user)
    else:
        # Static concurrency
        for _ in range(concurrency):
            task = asyncio.create_task(user_session(client, prompt_factory, stats, config, end_time, client_conf))
            tasks.append(task)

    # Wait for completion
    await asyncio.gather(*tasks)
    stats.stop_test()

    # Reports
    summary = stats.calculate_summary()

    # Push final summary to Prometheus if enabled
    if prom_exporter:
        final_summary = {
            'total_requests': float(summary.total_requests),
            'successful_requests': float(summary.successful_requests),
            'failed_requests': float(summary.failed_requests),
            'total_duration': summary.total_duration,
            'total_tokens': float(summary.total_tokens),
            'rps': summary.rps,
            'tps': summary.global_throughput_tokens_per_sec,
            'ttft_p50': summary.ttft_p50,
            'ttft_p90': summary.ttft_p90,
            'ttft_p99': summary.ttft_p99,
            'latency_p50': summary.latency_p50,
            'latency_p90': summary.latency_p90,
            'latency_p99': summary.latency_p99
        }
        prom_exporter.push_final_summary(final_summary)
        logger.info("Final test summary pushed to Prometheus")
    
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

    # Analysis Verdicts
    print(f"\n{Fore.MAGENTA}=== Automatic Analysis ===")
    verdicts = stats.analyze_results(summary)
    for v in verdicts:
        color = Fore.GREEN
        if "CRITICAL" in v:
            color = Fore.RED
        elif "WARNING" in v:
            color = Fore.YELLOW
        print(f"{color}{v}")

    # Per-Endpoint Breakdown (for Mixed Warfare)
    if "servers" in config:
        print(f"\n{Fore.CYAN}=== Per-Endpoint Breakdown ===")
        endpoint_stats = {}

        for m in stats.metrics:
            endpoint = m.endpoint or "unknown"
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    'successful': [],
                    'failed': []
                }

            if m.error is None:
                endpoint_stats[endpoint]['successful'].append(m)
            else:
                endpoint_stats[endpoint]['failed'].append(m)

        # Display stats for each endpoint
        for endpoint_name, data in sorted(endpoint_stats.items()):
            successful = data['successful']
            failed = data['failed']
            total = len(successful) + len(failed)

            print(f"\n{Fore.YELLOW}  {endpoint_name}:")
            print(f"    Total Requests: {total}")
            print(f"    Successful: {len(successful)}")
            print(f"    Failed: {len(failed)}")

            if successful:
                latencies = [m.latency_seconds for m in successful]
                ttfts = [m.ttft for m in successful]
                tokens = sum(m.output_tokens for m in successful)

                import numpy as np
                print(f"    Latency P50/P90/P99: {np.percentile(latencies, 50):.3f}s / "
                      f"{np.percentile(latencies, 90):.3f}s / {np.percentile(latencies, 99):.3f}s")
                print(f"    TTFT P50/P90/P99: {np.percentile(ttfts, 50):.3f}s / "
                      f"{np.percentile(ttfts, 90):.3f}s / {np.percentile(ttfts, 99):.3f}s")
                print(f"    Total Tokens: {tokens}")
                print(f"    Throughput: {tokens / summary.total_duration:.2f} tokens/sec")

    # Save details
    detailed_metrics = [
        {
            "request_id": m.request_id,
            "endpoint": m.endpoint,
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
