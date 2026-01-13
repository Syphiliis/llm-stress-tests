from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import numpy as np

@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    end_time: float = 0.0
    ttft: float = 0.0  # Time to first token
    output_tokens: int = 0
    error: Optional[str] = None
    endpoint: Optional[str] = None  # Track which endpoint was used (for mixed warfare)

    @property
    def latency_seconds(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        duration = self.latency_seconds
        if duration <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / duration

@dataclass
class TestSummary:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    total_tokens: int
    rps: float
    global_throughput_tokens_per_sec: float
    
    # Latency percentiles
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    
    latency_p50: float
    latency_p90: float
    latency_p99: float

class StatsCalculator:
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.start_time = 0.0
        self.end_time = 0.0

        # For incremental tracking
        self._last_snapshot_count = 0
        self._last_snapshot_tokens = 0
        self._last_snapshot_errors = 0

    def start_test(self):
        self.start_time = time.time()

    def stop_test(self):
        self.end_time = time.time()

    def add_metric(self, metric: RequestMetrics):
        self.metrics.append(metric)

    def calculate_summary(self) -> TestSummary:
        if not self.end_time:
            self.end_time = time.time()
            
        total_duration = self.end_time - self.start_time
        successful = [m for m in self.metrics if m.error is None]
        failed = [m for m in self.metrics if m.error is not None]
        
        total_tokens = sum(m.output_tokens for m in successful)
        
        # Avoid division by zero
        if total_duration > 0:
            rps = len(successful) / total_duration
            global_tps = total_tokens / total_duration
        else:
            rps = 0
            global_tps = 0

        # Percentiles
        if successful:
            ttfts = [m.ttft for m in successful]
            latencies = [m.latency_seconds for m in successful]
            
            ttft_p50 = float(np.percentile(ttfts, 50))
            ttft_p90 = float(np.percentile(ttfts, 90))
            ttft_p99 = float(np.percentile(ttfts, 99))
            
            lat_p50 = float(np.percentile(latencies, 50))
            lat_p90 = float(np.percentile(latencies, 90))
            lat_p99 = float(np.percentile(latencies, 99))
        else:
            ttft_p50 = ttft_p90 = ttft_p99 = 0.0
            lat_p50 = lat_p90 = lat_p99 = 0.0

        return TestSummary(
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration=total_duration,
            total_tokens=total_tokens,
            rps=rps,
            global_throughput_tokens_per_sec=global_tps,
            ttft_p50=ttft_p50,
            ttft_p90=ttft_p90,
            ttft_p99=ttft_p99,
            latency_p50=lat_p50,
            latency_p90=lat_p90,
            latency_p99=lat_p99
        )

    def analyze_results(self, summary: TestSummary) -> List[str]:
        verdicts = []

        # 1. Error Rate Analysis
        if summary.total_requests > 0:
            error_rate = summary.failed_requests / summary.total_requests
            if error_rate > 0.10:
                verdicts.append(f"CRITICAL: High Failure Rate ({error_rate:.1%}). System is unstable.")
            elif error_rate > 0.01:
                verdicts.append(f"WARNING: Non-zero Failure Rate ({error_rate:.1%}). Investigate errors.")
            else:
                verdicts.append("PASS: Error rate is acceptable (0%).")

        # 2. Latency Analysis (P90)
        # Thresholds: < 5s (Good), > 30s (Critical) for heavy load
        if summary.latency_p90 > 30.0:
            verdicts.append(f"CRITICAL: P90 Latency is extremely high ({summary.latency_p90:.2f}s). System is overloaded.")
        elif summary.latency_p90 > 5.0:
            verdicts.append(f"WARNING: P90 Latency is high ({summary.latency_p90:.2f}s). User experience degraded.")
        else:
            verdicts.append(f"PASS: P90 Latency is good ({summary.latency_p90:.2f}s).")

        # 3. TTFT Analysis (P50)
        # Thresholds: < 0.5s (Excellent), < 2s (Acceptable)
        if summary.ttft_p50 > 2.0:
            verdicts.append(f"CRITICAL: Time To First Token (P50) is slow ({summary.ttft_p50:.2f}s). Model is struggling.")
        elif summary.ttft_p50 < 0.5:
             verdicts.append(f"PASS: TTFT is excellent ({summary.ttft_p50:.2f}s).")
        else:
             verdicts.append(f"WARNING: TTFT is acceptable but could be better ({summary.ttft_p50:.2f}s).")

        return verdicts

    def get_current_snapshot(self) -> Dict[str, float]:
        """
        Calculate current metrics snapshot without stopping the test.
        Used for real-time Prometheus updates.

        Returns:
            Dictionary with current metric values
        """
        if not self.metrics:
            return {
                'latency_p50': 0.0,
                'latency_p90': 0.0,
                'latency_p99': 0.0,
                'ttft_p50': 0.0,
                'ttft_p90': 0.0,
                'ttft_p99': 0.0,
                'tps': 0.0,
                'rps': 0.0,
                'error_rate': 0.0,
                'contention_error_rate': 0.0
            }

        # Calculate elapsed time from start
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 1.0
        if elapsed_time <= 0:
            elapsed_time = 1.0

        successful = [m for m in self.metrics if m.error is None]
        failed = [m for m in self.metrics if m.error is not None]

        # Calculate error rates
        total_count = len(self.metrics)
        error_rate = len(failed) / total_count if total_count > 0 else 0.0

        # Contention errors: timeouts, 503s, connection errors
        contention_errors = [
            m for m in failed
            if m.error and any(
                keyword in m.error.lower()
                for keyword in ['timeout', '503', 'unavailable', 'connection']
            )
        ]
        contention_error_rate = len(contention_errors) / total_count if total_count > 0 else 0.0

        # Calculate throughput
        total_tokens = sum(m.output_tokens for m in successful)
        tps = total_tokens / elapsed_time
        rps = len(successful) / elapsed_time

        # Calculate percentiles
        if successful:
            ttfts = [m.ttft for m in successful]
            latencies = [m.latency_seconds for m in successful]

            ttft_p50 = float(np.percentile(ttfts, 50))
            ttft_p90 = float(np.percentile(ttfts, 90))
            ttft_p99 = float(np.percentile(ttfts, 99))

            lat_p50 = float(np.percentile(latencies, 50))
            lat_p90 = float(np.percentile(latencies, 90))
            lat_p99 = float(np.percentile(latencies, 99))
        else:
            ttft_p50 = ttft_p90 = ttft_p99 = 0.0
            lat_p50 = lat_p90 = lat_p99 = 0.0

        return {
            'latency_p50': lat_p50,
            'latency_p90': lat_p90,
            'latency_p99': lat_p99,
            'ttft_p50': ttft_p50,
            'ttft_p90': ttft_p90,
            'ttft_p99': ttft_p99,
            'tps': tps,
            'rps': rps,
            'error_rate': error_rate,
            'contention_error_rate': contention_error_rate
        }

    def get_delta_counts(self) -> Dict[str, int]:
        """
        Get incremental counts since last snapshot.
        Used for Prometheus Counter metrics.

        Returns:
            Dictionary with delta counts for requests, errors, and tokens
        """
        current_total = len(self.metrics)
        successful = [m for m in self.metrics if m.error is None]
        failed = [m for m in self.metrics if m.error is not None]

        current_tokens = sum(m.output_tokens for m in successful)
        current_errors = len(failed)

        deltas = {
            'requests': current_total - self._last_snapshot_count,
            'errors': current_errors - self._last_snapshot_errors,
            'tokens': current_tokens - self._last_snapshot_tokens
        }

        # Update last snapshot values
        self._last_snapshot_count = current_total
        self._last_snapshot_errors = current_errors
        self._last_snapshot_tokens = current_tokens

        return deltas
