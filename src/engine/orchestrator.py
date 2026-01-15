import asyncio
import time
import logging
import os
import json
from datetime import datetime as dt
from typing import List, Optional
import aiohttp
from colorama import Fore

from src.config.schema import GlobalConfig
from src.metrics.stats import StatsCalculator, TestSummary
from src.generators.prompt_factory import PromptFactory
from src.clients.base import BaseLLMClient
from src.clients.llama_cpp import LlamaCppClient
from src.clients.composite import WeightedCompositeClient

try:
    from src.metrics.prometheus_exporter import PrometheusExporter
except ImportError:
    PrometheusExporter = None

logger = logging.getLogger(__name__)

class LoadTestOrchestrator:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.stats = StatsCalculator()
        self.stop_event = asyncio.Event()
        self.active_users_count = 0
        self.active_users_lock = asyncio.Lock()
        
        # Initialize Prompt Factory
        self.prompt_factory = PromptFactory(
            seed=config.workload.seed or 42,
            min_tokens=config.prompts.min_tokens,
            max_tokens=config.prompts.max_tokens,
            prefix=config.prompts.prefix
        )

        # Initialize Client
        self.client = self._build_client()

        # Initialize Prometheus
        self.prom_exporter = None
        if config.prometheus.enabled:
            if PrometheusExporter:
                self.prom_exporter = PrometheusExporter(
                    pushgateway_url=config.prometheus.pushgateway_url,
                    job_name=config.prometheus.job_name,
                    instance_name=config.prometheus.instance_name,
                    username=config.prometheus.username,
                    password=config.prometheus.password
                )
                logger.info(f"Prometheus integration enabled: {config.prometheus.pushgateway_url}")
            else:
                logger.warning("Prometheus config enabled but library not found/loadable.")

    def _build_client(self) -> BaseLLMClient:
        servers = self.config.get_servers()
        if len(servers) == 1:
            return LlamaCppClient(servers[0])
        else:
            weighted_clients = []
            for s in servers:
                client = LlamaCppClient(s)
                weighted_clients.append((client, s.weight))
            return WeightedCompositeClient(weighted_clients)

    async def _user_session(self, session: aiohttp.ClientSession, end_time: float):
        async with self.active_users_lock:
            self.active_users_count += 1
        
        try:
            while time.time() < end_time and not self.stop_event.is_set():
                prompt = self.prompt_factory.generate_prompt()
                
                # Send request
                metric = await self.client.send_request(
                    session, 
                    prompt, 
                    self.config.prompts.max_tokens, 
                    self.config.client
                )
                
                self.stats.add_metric(metric)
                # Optional think time could go here
        finally:
            async with self.active_users_lock:
                self.active_users_count -= 1

    async def _progress_logger(self, start_time: float, end_time: float):
        interval = 10.0
        while time.time() < end_time and not self.stop_event.is_set():
            await asyncio.sleep(interval)
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            
            snapshot = self.stats.get_current_snapshot()
            async with self.active_users_lock:
                current_active = self.active_users_count

            # Format metrics with N/A handling
            lat_p90 = snapshot.get('latency_p90')
            lat_str = f"{lat_p90:.2f}s" if lat_p90 is not None else "N/A"
            
            print(f"\n{Fore.BLUE}[Progress @ {elapsed:.0f}s] "
                  f"Active Users: {current_active} | "
                  f"Requests: {len(self.stats.metrics)} | "
                  f"RPS: {snapshot['rps']:.1f} | "
                  f"TPS: {snapshot['tps']:.1f} tok/s | "
                  f"P90 Latency: {lat_str} | "
                  f"Errors: {snapshot['error_rate']:.1%} | "
                  f"Time Left: {max(0, remaining):.0f}s")

    async def _metrics_pusher(self, end_time: float):
        if not self.prom_exporter:
            return
            
        interval = self.config.prometheus.push_interval_seconds
        while time.time() < end_time and not self.stop_event.is_set():
            await asyncio.sleep(interval)
            
            snapshot = self.stats.get_current_snapshot()
            deltas = self.stats.get_delta_counts()
            
            async with self.active_users_lock:
                current_active = self.active_users_count

            self.prom_exporter.update_metrics(snapshot, current_active)
            self.prom_exporter.increment_counters(
                requests_delta=deltas['requests'],
                errors_delta=deltas['errors'],
                tokens_delta=deltas['tokens']
            )
            
            try:
                if self.prom_exporter.push_metrics():
                    logger.debug("Metrics pushed to Prometheus")
            except Exception as e:
                logger.error(f"Failed to push metrics: {e}")

    async def run(self):
        duration = self.config.workload.duration_seconds
        users = self.config.workload.users
        ramp_up = self.config.workload.ramp_up_seconds
        
        start_time = time.time()
        end_time = start_time + duration
        
        print(f"{Fore.CYAN}Starting load test with {users} users for {duration}s...")
        if self.config.is_mixed_warfare:
            print(f"{Fore.CYAN}Mode: Mixed Warfare ({len(self.config.servers)} endpoints)")
        else:
            print(f"{Fore.CYAN}Target: {self.config.server.base_url}")

        self.stats.start_test()
        
        # Configure overall client timeout
        timeout = aiohttp.ClientTimeout(
            total=self.config.client.timeout_seconds,
            connect=self.config.client.connect_timeout
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            
            # Background tasks
            tasks.append(asyncio.create_task(self._progress_logger(start_time, end_time)))
            if self.prom_exporter:
                tasks.append(asyncio.create_task(self._metrics_pusher(end_time)))

            # User tasks
            if ramp_up > 0:
                delay = ramp_up / users
                for i in range(users):
                    tasks.append(asyncio.create_task(self._user_session(session, end_time)))
                    if i < users - 1:
                        await asyncio.sleep(delay)
            else:
                for _ in range(users):
                    tasks.append(asyncio.create_task(self._user_session(session, end_time)))
            
            await asyncio.gather(*tasks)

        self.stats.stop_test()
        return self.stats

    def save_report(self, output_dir: str):
        summary = self.stats.calculate_summary()
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        
        # Print Console Summary
        self._print_summary(summary)
        
        # Save JSON with enhanced metrics
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
                "error": m.error,
                "error_category": m.error_category.value if m.error_category else None
            }
            for m in self.stats.metrics
        ]
        
        json_path = os.path.join(output_dir, f"test_run_{timestamp}.json")
        try:
            with open(json_path, "w") as f:
                json.dump({
                    "config": self.config.model_dump(),
                    "summary": {
                        "total_requests": summary.total_requests,
                        "successful_requests": summary.successful_requests,
                        "failed_requests": summary.failed_requests,
                        "is_failed": summary.is_failed,
                        "duration_seconds": summary.total_duration,
                        "total_tokens": summary.total_tokens,
                        "rps": summary.rps,
                        "tps": summary.global_throughput_tokens_per_sec,
                        # Latency metrics (null for N/A)
                        "latency_p50": summary.latency_p50,
                        "latency_p90": summary.latency_p90,
                        "latency_p99": summary.latency_p99,
                        "latency_mean": summary.latency_mean,
                        "latency_stdev": summary.latency_stdev,
                        # TTFT metrics (null for N/A)
                        "ttft_p50": summary.ttft_p50,
                        "ttft_p90": summary.ttft_p90,
                        "ttft_p99": summary.ttft_p99,
                        "ttft_mean": summary.ttft_mean,
                        "ttft_stdev": summary.ttft_stdev,
                        # Error breakdown
                        "error_breakdown": summary.error_breakdown
                    }, 
                    "details": detailed_metrics
                }, f, indent=2)
            print(f"\n{Fore.BLUE}Detailed logs saved to: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")

        # Final Prometheus Push (handle Optional values)
        if self.prom_exporter:
            # Only push non-None values to Prometheus
            summary_dict = {
                'total_requests': float(summary.total_requests),
                'successful_requests': float(summary.successful_requests),
                'failed_requests': float(summary.failed_requests),
                'total_duration': summary.total_duration,
                'total_tokens': float(summary.total_tokens),
                'rps': summary.rps,
                'tps': summary.global_throughput_tokens_per_sec,
            }
            # Add optional metrics only if available
            if summary.ttft_p50 is not None:
                summary_dict['ttft_p50'] = summary.ttft_p50
            if summary.ttft_p90 is not None:
                summary_dict['ttft_p90'] = summary.ttft_p90
            if summary.ttft_p99 is not None:
                summary_dict['ttft_p99'] = summary.ttft_p99
            if summary.latency_p50 is not None:
                summary_dict['latency_p50'] = summary.latency_p50
            if summary.latency_p90 is not None:
                summary_dict['latency_p90'] = summary.latency_p90
            if summary.latency_p99 is not None:
                summary_dict['latency_p99'] = summary.latency_p99
            if summary.latency_stdev is not None:
                summary_dict['latency_stdev'] = summary.latency_stdev
            if summary.ttft_stdev is not None:
                summary_dict['ttft_stdev'] = summary.ttft_stdev
                
            try:
                self.prom_exporter.push_final_summary(summary_dict)
                logger.info("Final test summary pushed to Prometheus")
            except Exception as e:
                logger.error(f"Failed to push final summary: {e}")

    def _print_summary(self, summary: TestSummary):
        """
        Print test summary with anti-hallucination support.
        Displays 'N/A' for missing metrics instead of misleading zeros.
        """
        # Helper function to format optional values
        def fmt(val, decimals=3, suffix=""):
            if val is None:
                return "N/A"
            return f"{val:.{decimals}f}{suffix}"
        
        print(f"\n{Fore.GREEN}=== Test Summary ===")
        
        # Check for complete failure
        if summary.is_failed:
            print(f"{Fore.RED}STATUS: FAILED (100% error rate)")
        
        print(f"Total Requests: {summary.total_requests}")
        print(f"Successful: {summary.successful_requests}")
        print(f"Failed: {summary.failed_requests}")
        print(f"Duration: {summary.total_duration:.2f}s")
        print(f"RPS: {summary.rps:.2f}")
        print(f"Global Throughput: {summary.global_throughput_tokens_per_sec:.2f} tokens/sec")
        
        # Latency section with stdev
        print(f"\n{Fore.YELLOW}=== Latency (s) ===")
        print(f"P50: {fmt(summary.latency_p50)}")
        print(f"P90: {fmt(summary.latency_p90)}")
        print(f"P99: {fmt(summary.latency_p99)}")
        print(f"Mean: {fmt(summary.latency_mean)}")
        print(f"Stdev (Jitter): {fmt(summary.latency_stdev)}")
        
        # TTFT section with stdev
        print(f"\n{Fore.YELLOW}=== TTFT (s) ===")
        print(f"P50: {fmt(summary.ttft_p50)}")
        print(f"P90: {fmt(summary.ttft_p90)}")
        print(f"P99: {fmt(summary.ttft_p99)}")
        print(f"Mean: {fmt(summary.ttft_mean)}")
        print(f"Stdev (Jitter): {fmt(summary.ttft_stdev)}")
        
        # Error breakdown
        if summary.error_breakdown and summary.failed_requests > 0:
            print(f"\n{Fore.RED}=== Error Breakdown ===")
            for category, count in summary.error_breakdown.items():
                if count > 0:
                    pct = count / summary.failed_requests * 100
                    print(f"  {category}: {count} ({pct:.1f}%)")
        
        if self.config.is_mixed_warfare:
            print(f"\n{Fore.CYAN}=== Per-Endpoint Breakdown ===")
            endpoint_stats = {}

            for m in self.stats.metrics:
                endpoint = m.endpoint or "unknown"
                if endpoint not in endpoint_stats:
                    endpoint_stats[endpoint] = {'successful': [], 'failed': []}

                if m.error is None:
                    endpoint_stats[endpoint]['successful'].append(m)
                else:
                    endpoint_stats[endpoint]['failed'].append(m)

            for endpoint_name, data in sorted(endpoint_stats.items()):
                successful = data['successful']
                failed = data['failed']
                total = len(successful) + len(failed)
                
                # Check if endpoint completely failed
                is_endpoint_failed = total > 0 and len(successful) == 0

                print(f"\n{Fore.YELLOW}  {endpoint_name}:")
                if is_endpoint_failed:
                    print(f"{Fore.RED}    STATUS: FAILED (100% error rate)")
                print(f"    Total Requests: {total}")
                print(f"    Successful: {len(successful)}")
                print(f"    Failed: {len(failed)}")

                if successful:
                    from src.metrics.stats import safe_percentile, safe_stdev
                    latencies = [m.latency_seconds for m in successful]
                    ttfts = [m.ttft for m in successful]
                    tokens = sum(m.output_tokens for m in successful)
                    
                    lat_p50 = safe_percentile(latencies, 50)
                    lat_p90 = safe_percentile(latencies, 90)
                    lat_p99 = safe_percentile(latencies, 99)
                    lat_stdev = safe_stdev(latencies)
                    
                    ttft_p50 = safe_percentile(ttfts, 50)
                    ttft_p90 = safe_percentile(ttfts, 90)
                    ttft_p99 = safe_percentile(ttfts, 99)
                    
                    print(f"    Latency P50/P90/P99: {fmt(lat_p50)}s / {fmt(lat_p90)}s / {fmt(lat_p99)}s")
                    print(f"    Latency Stdev: {fmt(lat_stdev)}s")
                    print(f"    TTFT P50/P90/P99: {fmt(ttft_p50)}s / {fmt(ttft_p90)}s / {fmt(ttft_p99)}s")
                    print(f"    Total Tokens: {tokens}")
                else:
                    print(f"    Latency: N/A (no successful requests)")
                    print(f"    TTFT: N/A (no successful requests)")
                
                # Show error breakdown for this endpoint
                if failed:
                    from src.metrics.stats import categorize_error, ErrorCategory
                    error_counts = {cat.value: 0 for cat in ErrorCategory}
                    for m in failed:
                        cat = categorize_error(m.error)
                        error_counts[cat.value] += 1
                    
                    errors_str = ", ".join(f"{k}: {v}" for k, v in error_counts.items() if v > 0)
                    print(f"    Errors: {errors_str}")
