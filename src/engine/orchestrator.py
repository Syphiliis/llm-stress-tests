import asyncio
import time
import logging
import os
import json
import csv
from datetime import datetime as dt
from typing import List, Optional
import aiohttp
from colorama import Fore
from urllib.parse import urlparse

from src.config.schema import GlobalConfig
from src.metrics.stats import StatsCalculator, TestSummary
from src.generators.prompt_factory import PromptFactory
from src.generators.token_scheduler import TokenScheduler
from src.clients.base import BaseLLMClient
from src.clients.llama_cpp import LlamaCppClient
from src.clients.composite import WeightedCompositeClient
from src.metrics.network import PingMonitor
from src.metrics.system_sampler import SystemSampler

try:
    from src.metrics.prometheus_exporter import PrometheusExporter
except ImportError:
    PrometheusExporter = None

logger = logging.getLogger(__name__)

class LoadTestOrchestrator:
    CSV_COLUMNS = [
        "record_type",
        "run_id",
        "ts",
        "label",
        "model",
        "iteration",
        "request_id",
        "endpoint",
        "user_id",
        "start_ts",
        "end_ts",
        "ttft",
        "latency",
        "queue_wait",
        "concurrency",
        "input_tokens",
        "output_tokens",
        "tps",
        "latency_per_token",
        "error",
        "error_category",
        "ping_min_ms",
        "ping_avg_ms",
        "ping_max_ms",
        "ping_loss_pct",
        "cpu_avg",
        "cpu_per_core",
        "ram_used_mb",
        "ram_percent",
        "gpu",
        "metric_name",
        "metric_value",
        "metric_p50",
        "metric_p90",
        "metric_p99",
        "count",
        "input_bucket",
        "summary_total_requests",
        "summary_successful_requests",
        "summary_failed_requests",
        "summary_duration_seconds",
        "summary_total_tokens",
        "summary_rps",
        "summary_tps",
        "summary_latency_p50",
        "summary_latency_p90",
        "summary_latency_p99",
        "summary_ttft_p50",
        "summary_ttft_p90",
        "summary_ttft_p99",
        "summary_queue_wait_p50",
        "summary_queue_wait_p90",
        "summary_per_request_tps_p50",
        "summary_per_request_tps_p90",
        "summary_latency_per_token_mean",
        "summary_latency_per_token_p90",
        "summary_latency_slope_vs_input",
        "summary_stability_score",
        "summary_error_breakdown",
        "summary_per_user_throughput",
        "summary_tokens_vs_concurrency",
        "bottleneck_hint",
        "config_json",
    ]

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
        self.token_scheduler = TokenScheduler(config.prompts, config.workload.duration_seconds)

        # Initialize Client
        self.client = self._build_client()
        self.ping_monitor: Optional[PingMonitor] = None
        self.system_sampler: Optional[SystemSampler] = None

        try:
            first_server = self.config.get_servers()[0]
            parsed = urlparse(first_server.base_url)
            self.ping_host = parsed.hostname
        except Exception:
            self.ping_host = None

        if config.ping.enabled and self.ping_host:
            self.ping_monitor = PingMonitor(
                host=self.ping_host,
                count=config.ping.count,
                interval_seconds=config.ping.interval_seconds
            )
        if config.system.enabled:
            self.system_sampler = SystemSampler(
                interval_seconds=config.system.interval_seconds,
                gpu_command=config.system.gpu_command
            )

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

    async def _user_session(self, session: aiohttp.ClientSession, end_time: float, user_id: int):
        async with self.active_users_lock:
            self.active_users_count += 1
        
        try:
            while time.time() < end_time and not self.stop_event.is_set():
                elapsed = time.time() - self.stats.start_time
                target_tokens = self.token_scheduler.target_tokens(elapsed)
                input_tokens, prompt = self.prompt_factory.generate_prompt_with_size(target_tokens)
                async with self.active_users_lock:
                    current_active = self.active_users_count
                
                # Send request
                scheduled_time = time.time()
                metric = await self.client.send_request(
                    session, 
                    prompt, 
                    self.config.prompts.max_tokens, 
                    self.config.client,
                    input_tokens=input_tokens
                )
                metric.queue_wait = metric.start_time - scheduled_time
                metric.concurrency = current_active
                metric.user_id = user_id
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
        if self.ping_monitor:
            await self.ping_monitor.run_once()  # baseline before load
            self.ping_monitor.start()
        if self.system_sampler:
            self.system_sampler.start()
        
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
                    tasks.append(asyncio.create_task(self._user_session(session, end_time, user_id=i + 1)))
                    if i < users - 1:
                        await asyncio.sleep(delay)
            else:
                for i in range(users):
                    tasks.append(asyncio.create_task(self._user_session(session, end_time, user_id=i + 1)))
            
            await asyncio.gather(*tasks)

        self.stats.stop_test()
        if self.ping_monitor:
            await self.ping_monitor.stop()
        if self.system_sampler:
            await self.system_sampler.stop()
        return self.stats

    def save_report(
        self,
        output_dir: str,
        label: Optional[str] = None,
        csv_path: Optional[str] = None,
        include_header: bool = True,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None
    ):
        summary = self.stats.calculate_summary()
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        model_name = self.config.server.name if self.config.server else "default"
        run_id = run_id or f"{model_name}_{timestamp}{suffix}"
        
        # Print Console Summary
        self._print_summary(summary)
        
        detailed_metrics = [
            {
                "record_type": "request",
                "run_id": run_id,
                "ts": m.start_time,
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "request_id": m.request_id,
                "endpoint": m.endpoint,
                "user_id": m.user_id,
                "start_ts": m.start_time,
                "end_ts": m.end_time,
                "ttft": m.ttft,
                "latency": m.latency_seconds,
                "queue_wait": m.queue_wait,
                "concurrency": m.concurrency,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "tps": m.tokens_per_second,
                "latency_per_token": m.latency_per_token,
                "error": m.error,
                "error_category": m.error_category.value if m.error_category else None
            }
            for m in self.stats.metrics
        ]

        bottleneck_hint = self._infer_bottleneck(summary)

        summary_row = {
            "record_type": "summary",
            "run_id": run_id,
            "ts": time.time(),
            "label": label,
            "model": model_name,
            "iteration": iteration,
            "summary_total_requests": summary.total_requests,
            "summary_successful_requests": summary.successful_requests,
            "summary_failed_requests": summary.failed_requests,
            "summary_duration_seconds": summary.total_duration,
            "summary_total_tokens": summary.total_tokens,
            "summary_rps": summary.rps,
            "summary_tps": summary.global_throughput_tokens_per_sec,
            "summary_latency_p50": summary.latency_p50,
            "summary_latency_p90": summary.latency_p90,
            "summary_latency_p99": summary.latency_p99,
            "summary_ttft_p50": summary.ttft_p50,
            "summary_ttft_p90": summary.ttft_p90,
            "summary_ttft_p99": summary.ttft_p99,
            "summary_queue_wait_p50": summary.queue_wait_p50,
            "summary_queue_wait_p90": summary.queue_wait_p90,
            "summary_per_request_tps_p50": summary.per_request_tps_p50,
            "summary_per_request_tps_p90": summary.per_request_tps_p90,
            "summary_latency_per_token_mean": summary.latency_per_token_mean,
            "summary_latency_per_token_p90": summary.latency_per_token_p90,
            "summary_latency_slope_vs_input": summary.latency_slope_vs_input,
            "summary_stability_score": summary.stability_score,
            "summary_error_breakdown": json.dumps(summary.error_breakdown) if summary.error_breakdown else None,
            "summary_per_user_throughput": json.dumps(summary.per_user_throughput) if summary.per_user_throughput else None,
            "summary_tokens_vs_concurrency": json.dumps(summary.tokens_vs_concurrency) if summary.tokens_vs_concurrency else None,
            "bottleneck_hint": bottleneck_hint,
        }

        config_row = {
            "record_type": "config",
            "run_id": run_id,
            "ts": time.time(),
            "label": label,
            "model": model_name,
            "iteration": iteration,
            "config_json": json.dumps(self.config.model_dump()),
        }

        ping_rows = []
        if self.ping_monitor and self.ping_monitor.samples:
            for s in self.ping_monitor.samples:
                ping_rows.append({
                    "record_type": "ping",
                    "run_id": run_id,
                    "ts": s.get("ts"),
                    "label": label,
                    "model": model_name,
                    "iteration": iteration,
                    "ping_min_ms": s.get("min_ms"),
                    "ping_avg_ms": s.get("avg_ms"),
                    "ping_max_ms": s.get("max_ms"),
                    "ping_loss_pct": s.get("loss_pct"),
                })

        system_rows = []
        if self.system_sampler and self.system_sampler.snapshots:
            for s in self.system_sampler.snapshots:
                system_rows.append({
                    "record_type": "system",
                    "run_id": run_id,
                    "ts": s.get("ts"),
                    "label": label,
                    "model": model_name,
                    "iteration": iteration,
                    "cpu_avg": s.get("cpu_avg"),
                    "cpu_per_core": json.dumps(s.get("cpu_per_core")),
                    "ram_used_mb": s.get("ram_used_mb"),
                    "ram_percent": s.get("ram_percent"),
                    "gpu": json.dumps(s.get("gpu")),
                })

        ttft_rows = self._ttft_vs_input_rows(run_id, label, model_name, iteration)
        tokens_vs_conc_rows = self._tokens_vs_concurrency_rows(run_id, label, model_name, iteration)
        reactivity_rows = self._reactivity_rows(run_id, label, model_name, iteration)
        per_user_rows = self._per_user_tps_rows(run_id, label, model_name, iteration)

        combined_rows = (
            detailed_metrics
            + [summary_row, config_row]
            + ping_rows
            + system_rows
            + ttft_rows
            + tokens_vs_conc_rows
            + reactivity_rows
            + per_user_rows
        )

        csv_path = csv_path or os.path.join(output_dir, f"test_run_{timestamp}{suffix}.csv")
        try:
            self._write_combined_csv(csv_path, combined_rows, include_header=include_header)
            print(f"{Fore.BLUE}Combined CSV saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save combined CSV: {e}")

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

    def _write_combined_csv(self, path: str, rows: List[dict], include_header: bool = True):
        if not rows:
            return
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            if include_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _ttft_vs_input_rows(self, run_id: str, label: Optional[str], model_name: str, iteration: Optional[int]) -> List[dict]:
        bucket_size = 256
        if self.config.prompts.stages:
            min_stage = min(s.tokens for s in self.config.prompts.stages)
            if min_stage > 0:
                bucket_size = min_stage

        buckets = {}
        for m in self.stats.metrics:
            if m.error or m.input_tokens <= 0:
                continue
            bucket = int(m.input_tokens / bucket_size) * bucket_size
            buckets.setdefault(bucket, []).append(m.ttft)

        rows = []
        from src.metrics.stats import safe_percentile
        for bucket, values in sorted(buckets.items()):
            rows.append({
                "record_type": "ttft_vs_input",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "input_bucket": bucket,
                "metric_name": "ttft",
                "metric_p50": safe_percentile(values, 50),
                "metric_p90": safe_percentile(values, 90),
                "metric_p99": safe_percentile(values, 99),
                "count": len(values)
            })
        return rows

    def _tokens_vs_concurrency_rows(self, run_id: str, label: Optional[str], model_name: str, iteration: Optional[int]) -> List[dict]:
        rows = []
        if not self.stats.metrics:
            return rows
        buckets = {}
        for m in self.stats.metrics:
            if m.error or m.concurrency <= 0:
                continue
            buckets.setdefault(m.concurrency, []).append(m.tokens_per_second)
        from src.metrics.stats import safe_mean
        for conc, values in sorted(buckets.items()):
            rows.append({
                "record_type": "tokens_vs_concurrency",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "concurrency": conc,
                "metric_name": "tps",
                "metric_value": safe_mean(values),
                "count": len(values)
            })
        return rows

    def _reactivity_rows(self, run_id: str, label: Optional[str], model_name: str, iteration: Optional[int]) -> List[dict]:
        successful = [m for m in self.stats.metrics if m.error is None]
        if len(successful) < 2:
            return []

        successful.sort(key=lambda m: m.start_time)
        window = max(1, int(len(successful) * 0.1))
        first = successful[:window]
        last = successful[-window:]

        from src.metrics.stats import safe_percentile
        ttft_initial = safe_percentile([m.ttft for m in first], 50)
        ttft_final = safe_percentile([m.ttft for m in last], 50)

        rows = [
            {
                "record_type": "reactivity",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "metric_name": "ttft_p50_initial",
                "metric_value": ttft_initial,
            },
            {
                "record_type": "reactivity",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "metric_name": "ttft_p50_final",
                "metric_value": ttft_final,
            }
        ]

        if ttft_initial is not None and ttft_final is not None:
            rows.append({
                "record_type": "reactivity",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "metric_name": "ttft_p50_delta",
                "metric_value": ttft_final - ttft_initial,
            })
            if ttft_initial > 0:
                rows.append({
                    "record_type": "reactivity",
                    "run_id": run_id,
                    "ts": time.time(),
                    "label": label,
                    "model": model_name,
                    "iteration": iteration,
                    "metric_name": "ttft_p50_ratio",
                    "metric_value": ttft_final / ttft_initial,
                })
        return rows

    def _infer_bottleneck(self, summary: TestSummary) -> str:
        ping_avg = None
        if self.ping_monitor and self.ping_monitor.samples:
            avgs = [s.get("avg_ms") for s in self.ping_monitor.samples if s.get("avg_ms") is not None]
            if avgs:
                ping_avg = sum(avgs) / len(avgs)

        cpu_avg = None
        ram_pct = None
        gpu_util = None
        if self.system_sampler and self.system_sampler.snapshots:
            cpu_vals = [s.get("cpu_avg") for s in self.system_sampler.snapshots if s.get("cpu_avg") is not None]
            ram_vals = [s.get("ram_percent") for s in self.system_sampler.snapshots if s.get("ram_percent") is not None]
            gpu_vals = []
            for s in self.system_sampler.snapshots:
                gpus = s.get("gpu") or []
                for g in gpus:
                    if g.get("utilization_gpu") is not None:
                        gpu_vals.append(g.get("utilization_gpu"))
            if cpu_vals:
                cpu_avg = sum(cpu_vals) / len(cpu_vals)
            if ram_vals:
                ram_pct = sum(ram_vals) / len(ram_vals)
            if gpu_vals:
                gpu_util = sum(gpu_vals) / len(gpu_vals)

        if ping_avg is not None and ping_avg > 100 and summary.ttft_p50 and summary.ttft_p50 > 2.0:
            return "network"
        if gpu_util is not None and gpu_util > 90:
            return "gpu"
        if cpu_avg is not None and cpu_avg > 90:
            return "cpu"
        if ram_pct is not None and ram_pct > 90:
            return "memory"
        if summary.queue_wait_p90 is not None and summary.queue_wait_p90 > 1.0:
            return "client_queue"
        if summary.latency_slope_vs_input is not None and summary.latency_slope_vs_input > 0.01:
            return "model_throughput"
        return "unknown"

    def _per_user_tps_rows(self, run_id: str, label: Optional[str], model_name: str, iteration: Optional[int]) -> List[dict]:
        if not self.stats.metrics:
            return []
        per_user = {}
        for m in self.stats.metrics:
            if m.error:
                continue
            uid = m.user_id if m.user_id is not None else "unknown"
            per_user.setdefault(uid, 0)
            per_user[uid] += m.output_tokens
        if not per_user or self.stats.start_time <= 0:
            return []
        duration = max(1e-6, self.stats.end_time - self.stats.start_time)
        rows = []
        for uid, tokens in per_user.items():
            rows.append({
                "record_type": "per_user_tps",
                "run_id": run_id,
                "ts": time.time(),
                "label": label,
                "model": model_name,
                "iteration": iteration,
                "user_id": uid,
                "metric_name": "tps_per_user",
                "metric_value": tokens / duration,
            })
        return rows

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
        
        # Print automated analysis verdicts
        verdicts = self.stats.analyze_results(summary)
        
        # Add capacity recommendation based on results
        users = self.config.workload.users
        if summary.latency_p90 is not None:
             if summary.latency_p90 > 30.0:
                 verdicts.append(f"{Fore.RED}CAPACITY: System overloaded with {users} users. Recommend reducing concurrency (start with {max(1, int(users*0.5))} users).")
             elif summary.latency_p90 < 2.0:
                 verdicts.append(f"{Fore.GREEN}CAPACITY: System handles {users} users comfortably. You can likely increase load.")
             else:
                 verdicts.append(f"{Fore.YELLOW}CAPACITY: System implies load is moderate/heavy at {users} users.")

        if verdicts:
            print(f"\n{Fore.MAGENTA}=== Automated Analysis ===")
            for v in verdicts:
                color = Fore.GREEN
                if "CRITICAL" in v or "CAPACITY" in v and "overloaded" in v: color = Fore.RED
                elif "WARNING" in v or "CAPACITY" in v and "moderate" in v: color = Fore.YELLOW
                elif "INFO" in v: color = Fore.BLUE
                elif "N/A" in v: color = Fore.LIGHTBLACK_EX
                elif "CAPACITY" in v and "comfortably" in v: color = Fore.GREEN
                print(f"{color}{v}")

        print(f"Total Requests: {summary.total_requests}")
        print(f"Successful: {summary.successful_requests}")
        print(f"Failed: {summary.failed_requests}")
        print(f"Duration: {summary.total_duration:.2f}s")
        print(f"RPS: {summary.rps:.2f}")
        print(f"Global Throughput: {summary.global_throughput_tokens_per_sec:.2f} tokens/sec")

        # Print Legend
        print(f"\n{Fore.WHITE}=== Legend ===")
        print(f"{Fore.LIGHTBLACK_EX}  P50 (Median)   : {Fore.WHITE}50% of requests were faster than this.")
        print(f"{Fore.LIGHTBLACK_EX}  P90            : {Fore.WHITE}90% of requests were faster than this (excludes worst outliers).")
        print(f"{Fore.LIGHTBLACK_EX}  TTFT           : {Fore.WHITE}Time To First Token (response reactiveness).")
        print(f"{Fore.LIGHTBLACK_EX}  Latency        : {Fore.WHITE}Total request processing time.")
        print(f"{Fore.LIGHTBLACK_EX}  Stdev (Jitter) : {Fore.WHITE}Stability metric (lower is better). High = unstable.")
        print(f"{Fore.LIGHTBLACK_EX}  RPS            : {Fore.WHITE}Requests Per Second (system throughput).")
        
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

        if summary.queue_wait_p50 is not None:
            print(f"\n{Fore.YELLOW}=== Queue Wait (s) ===")
            print(f"P50: {fmt(summary.queue_wait_p50)}")
            print(f"P90: {fmt(summary.queue_wait_p90)}")

        if summary.per_request_tps_p50 is not None:
            print(f"\n{Fore.YELLOW}=== Throughput Per Request (tokens/sec) ===")
            print(f"P50: {fmt(summary.per_request_tps_p50)}")
            print(f"P90: {fmt(summary.per_request_tps_p90)}")

        if summary.latency_per_token_mean is not None:
            print(f"\n{Fore.YELLOW}=== Latency Per Token (s/token) ===")
            print(f"Mean: {fmt(summary.latency_per_token_mean)}")
            print(f"P90: {fmt(summary.latency_per_token_p90)}")

        if summary.stability_score is not None:
            print(f"\n{Fore.YELLOW}Stability Score: {fmt(summary.stability_score, decimals=2)} (1.0=perfect, 0=unstable)")

        if summary.tokens_vs_concurrency:
            print(f"\n{Fore.CYAN}Tokens/sec vs Concurrency:")
            for conc, tps in sorted(summary.tokens_vs_concurrency.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 9999):
                if tps is None:
                    continue
                print(f"  Concurrency {conc}: {tps:.2f} tok/s")
        
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
