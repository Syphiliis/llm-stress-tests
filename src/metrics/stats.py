from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time
import numpy as np


class ErrorCategory(Enum):
    """Categorization of errors for anti-hallucination reporting."""
    TIMEOUT = "timeout"
    HTTP_5XX = "http_5xx"
    CONNECTION_REFUSED = "connection_refused"
    OTHER = "other"


def categorize_error(error_msg: Optional[str]) -> ErrorCategory:
    """
    Categorize an error message into a specific error type.
    
    Args:
        error_msg: The error message string
        
    Returns:
        ErrorCategory enum value
    """
    if not error_msg:
        return ErrorCategory.OTHER
    
    error_lower = error_msg.lower()
    
    # Timeout errors
    if any(kw in error_lower for kw in ['timeout', 'timed out', 'timedout']):
        return ErrorCategory.TIMEOUT
    
    # HTTP 5xx errors
    if any(kw in error_lower for kw in ['http 5', '500', '502', '503', '504', 'internal server error', 'bad gateway', 'service unavailable', 'gateway timeout']):
        return ErrorCategory.HTTP_5XX
    
    # Connection refused errors
    if any(kw in error_lower for kw in ['connection refused', 'connect error', 'cannot connect', 'connection reset', 'connection closed', 'refused']):
        return ErrorCategory.CONNECTION_REFUSED
    
    return ErrorCategory.OTHER


def safe_percentile(data: List[float], percentile: float) -> Optional[float]:
    """
    Safely calculate percentile, returning None for empty data.
    
    Args:
        data: List of float values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value or None if data is empty
    """
    if not data:
        return None
    try:
        return float(np.percentile(data, percentile))
    except (ValueError, IndexError):
        return None


def safe_mean(data: List[float]) -> Optional[float]:
    """
    Safely calculate mean, returning None for empty data.
    
    Args:
        data: List of float values
        
    Returns:
        Mean value or None if data is empty
    """
    if not data:
        return None
    try:
        return float(np.mean(data))
    except (ValueError, IndexError):
        return None


def safe_stdev(data: List[float]) -> Optional[float]:
    """
    Safely calculate standard deviation, returning None for insufficient data.
    
    Args:
        data: List of float values
        
    Returns:
        Standard deviation or None if data has fewer than 2 elements
    """
    if not data or len(data) < 2:
        return None
    try:
        return float(np.std(data, ddof=1))  # Sample stdev
    except (ValueError, IndexError):
        return None


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
    
    @property
    def error_category(self) -> Optional[ErrorCategory]:
        """Returns the categorized error type if there's an error."""
        if self.error:
            return categorize_error(self.error)
        return None


@dataclass
class TestSummary:
    """
    Test summary with anti-hallucination support.
    
    Metrics that are None indicate "N/A" (no data available).
    This prevents confusion between "0.00" (actual zero value) 
    and "no data" scenarios.
    """
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    total_tokens: int
    rps: float
    global_throughput_tokens_per_sec: float
    
    # TTFT metrics - None means N/A (no successful requests)
    ttft_p50: Optional[float] = None
    ttft_p90: Optional[float] = None
    ttft_p99: Optional[float] = None
    ttft_mean: Optional[float] = None
    ttft_stdev: Optional[float] = None  # Jitter measurement
    
    # Latency metrics - None means N/A
    latency_p50: Optional[float] = None
    latency_p90: Optional[float] = None
    latency_p99: Optional[float] = None
    latency_mean: Optional[float] = None
    latency_stdev: Optional[float] = None  # Jitter measurement
    
    # Error breakdown by category
    error_breakdown: Optional[Dict[str, int]] = None
    
    # Reliability flag: True if 100% failure rate
    is_failed: bool = False

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
            rps = 0.0
            global_tps = 0.0

        # Extract raw data for calculations
        ttfts = [m.ttft for m in successful] if successful else []
        latencies = [m.latency_seconds for m in successful] if successful else []
        
        # Calculate error breakdown by category
        error_breakdown: Dict[str, int] = {
            ErrorCategory.TIMEOUT.value: 0,
            ErrorCategory.HTTP_5XX.value: 0,
            ErrorCategory.CONNECTION_REFUSED.value: 0,
            ErrorCategory.OTHER.value: 0
        }
        
        for m in failed:
            category = categorize_error(m.error)
            error_breakdown[category.value] += 1
        
        # Check if test completely failed
        is_failed = len(self.metrics) > 0 and len(successful) == 0

        return TestSummary(
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration=total_duration,
            total_tokens=total_tokens,
            rps=rps,
            global_throughput_tokens_per_sec=global_tps,
            # TTFT metrics (None if no data)
            ttft_p50=safe_percentile(ttfts, 50),
            ttft_p90=safe_percentile(ttfts, 90),
            ttft_p99=safe_percentile(ttfts, 99),
            ttft_mean=safe_mean(ttfts),
            ttft_stdev=safe_stdev(ttfts),
            # Latency metrics (None if no data)
            latency_p50=safe_percentile(latencies, 50),
            latency_p90=safe_percentile(latencies, 90),
            latency_p99=safe_percentile(latencies, 99),
            latency_mean=safe_mean(latencies),
            latency_stdev=safe_stdev(latencies),
            # Error breakdown
            error_breakdown=error_breakdown if failed else None,
            is_failed=is_failed
        )

    def analyze_results(self, summary: TestSummary) -> List[str]:
        """
        Analyze test results and generate verdicts.
        Handles N/A (None) values for anti-hallucination reporting.
        """
        verdicts = []

        # 0. Check for complete failure
        if summary.is_failed:
            verdicts.append("CRITICAL: 100% Failure Rate. No successful requests. Test FAILED.")
            if summary.error_breakdown:
                verdicts.append(f"  Error Breakdown: {summary.error_breakdown}")
            return verdicts

        # 1. Error Rate Analysis
        if summary.total_requests > 0:
            error_rate = summary.failed_requests / summary.total_requests
            if error_rate > 0.10:
                verdicts.append(f"CRITICAL: High Failure Rate ({error_rate:.1%}). System is unstable.")
            elif error_rate > 0.01:
                verdicts.append(f"WARNING: Non-zero Failure Rate ({error_rate:.1%}). Investigate errors.")
            else:
                verdicts.append("PASS: Error rate is acceptable (<1%).")
            
            # Add error breakdown if errors exist
            if summary.error_breakdown and summary.failed_requests > 0:
                breakdown_str = ", ".join(f"{k}: {v}" for k, v in summary.error_breakdown.items() if v > 0)
                verdicts.append(f"  Error Categories: {breakdown_str}")

        # 2. Latency Analysis (P90)
        # Thresholds: < 5s (Good), > 30s (Critical) for heavy load
        if summary.latency_p90 is None:
            verdicts.append("N/A: P90 Latency unavailable (no successful requests).")
        elif summary.latency_p90 > 30.0:
            verdicts.append(f"CRITICAL: P90 Latency is extremely high ({summary.latency_p90:.2f}s). System is overloaded.")
        elif summary.latency_p90 > 5.0:
            verdicts.append(f"WARNING: P90 Latency is high ({summary.latency_p90:.2f}s). User experience degraded.")
        else:
            verdicts.append(f"PASS: P90 Latency is good ({summary.latency_p90:.2f}s).")

        # 3. TTFT Analysis (P50)
        # Thresholds: < 0.5s (Excellent), < 2s (Acceptable)
        if summary.ttft_p50 is None:
            verdicts.append("N/A: TTFT unavailable (no successful requests).")
        elif summary.ttft_p50 > 2.0:
            verdicts.append(f"CRITICAL: Time To First Token (P50) is slow ({summary.ttft_p50:.2f}s). Model is struggling.")
        elif summary.ttft_p50 < 0.5:
            verdicts.append(f"PASS: TTFT is excellent ({summary.ttft_p50:.2f}s).")
        else:
            verdicts.append(f"WARNING: TTFT is acceptable but could be better ({summary.ttft_p50:.2f}s).")

        # 4. Jitter Analysis (Stdev)
        if summary.latency_stdev is not None and summary.latency_mean is not None:
            cv = summary.latency_stdev / summary.latency_mean if summary.latency_mean > 0 else 0
            if cv > 0.5:
                verdicts.append(f"WARNING: High latency jitter (CV={cv:.2f}). Inconsistent response times.")
            elif cv > 0.3:
                verdicts.append(f"INFO: Moderate latency jitter (CV={cv:.2f}).")

        return verdicts

    def get_current_snapshot(self) -> Dict[str, Optional[float]]:
        """
        Calculate current metrics snapshot without stopping the test.
        Used for real-time Prometheus updates.

        Returns:
            Dictionary with current metric values. None indicates N/A.
        """
        if not self.metrics:
            return {
                'latency_p50': None,
                'latency_p90': None,
                'latency_p99': None,
                'latency_stdev': None,
                'ttft_p50': None,
                'ttft_p90': None,
                'ttft_p99': None,
                'ttft_stdev': None,
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

        # Extract data for calculations
        ttfts = [m.ttft for m in successful] if successful else []
        latencies = [m.latency_seconds for m in successful] if successful else []

        return {
            'latency_p50': safe_percentile(latencies, 50),
            'latency_p90': safe_percentile(latencies, 90),
            'latency_p99': safe_percentile(latencies, 99),
            'latency_stdev': safe_stdev(latencies),
            'ttft_p50': safe_percentile(ttfts, 50),
            'ttft_p90': safe_percentile(ttfts, 90),
            'ttft_p99': safe_percentile(ttfts, 99),
            'ttft_stdev': safe_stdev(ttfts),
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
