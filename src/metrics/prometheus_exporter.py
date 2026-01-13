import logging
import time
from typing import Optional, Dict, List
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from prometheus_client.exposition import basic_auth_handler

from src.metrics.stats import RequestMetrics

logger = logging.getLogger(__name__)

class PrometheusExporter:
    """
    Exports load test metrics to Prometheus Pushgateway in real-time.

    Key Metrics:
    - client_lag_p50/p90/p99: Latency percentiles
    - ttft_p50/p90/p99: Time to first token percentiles
    - tokens_per_second: Throughput (TPS)
    - request_rate: Requests per second
    - error_rate: Percentage of failed requests
    - contention_error_rate: Rate of timeout/503 errors indicating resource contention
    """

    def __init__(
        self,
        pushgateway_url: str,
        job_name: str = "llm_load_test",
        instance_name: str = "local",
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.instance_name = instance_name
        self.username = username
        self.password = password

        # Create a separate registry to avoid conflicts
        self.registry = CollectorRegistry()

        # Define metrics
        # Latency metrics (client lag)
        self.client_lag_p50 = Gauge(
            'llm_client_lag_p50_seconds',
            'P50 latency from client perspective',
            registry=self.registry
        )
        self.client_lag_p90 = Gauge(
            'llm_client_lag_p90_seconds',
            'P90 latency from client perspective',
            registry=self.registry
        )
        self.client_lag_p99 = Gauge(
            'llm_client_lag_p99_seconds',
            'P99 latency from client perspective',
            registry=self.registry
        )

        # TTFT metrics
        self.ttft_p50 = Gauge(
            'llm_ttft_p50_seconds',
            'P50 time to first token',
            registry=self.registry
        )
        self.ttft_p90 = Gauge(
            'llm_ttft_p90_seconds',
            'P90 time to first token',
            registry=self.registry
        )
        self.ttft_p99 = Gauge(
            'llm_ttft_p99_seconds',
            'P99 time to first token',
            registry=self.registry
        )

        # Throughput metrics
        self.tokens_per_second = Gauge(
            'llm_tokens_per_second',
            'Global throughput in tokens per second',
            registry=self.registry
        )

        # Request rate
        self.request_rate = Gauge(
            'llm_request_rate_per_second',
            'Requests per second',
            registry=self.registry
        )

        # Error rate metrics
        self.error_rate = Gauge(
            'llm_error_rate',
            'Percentage of failed requests',
            registry=self.registry
        )

        self.contention_error_rate = Gauge(
            'llm_contention_error_rate',
            'Rate of timeout/503 errors indicating resource contention',
            registry=self.registry
        )

        # Counters for total tracking
        self.total_requests = Counter(
            'llm_total_requests',
            'Total number of requests made',
            registry=self.registry
        )

        self.total_errors = Counter(
            'llm_total_errors',
            'Total number of failed requests',
            registry=self.registry
        )

        self.total_tokens = Counter(
            'llm_total_tokens',
            'Total number of tokens generated',
            registry=self.registry
        )

        # Test metadata
        self.active_users = Gauge(
            'llm_active_users',
            'Current number of active concurrent users',
            registry=self.registry
        )

        logger.info(f"Prometheus exporter initialized for {pushgateway_url}")

    def update_metrics(
        self,
        metrics_snapshot: Dict[str, float],
        active_users: int
    ):
        """
        Updates Prometheus metrics based on current test snapshot.

        Args:
            metrics_snapshot: Dictionary containing current metric values
                Expected keys: latency_p50, latency_p90, latency_p99,
                              ttft_p50, ttft_p90, ttft_p99, tps, rps,
                              error_rate, contention_error_rate
            active_users: Current number of active concurrent users
        """
        # Update latency metrics
        if 'latency_p50' in metrics_snapshot:
            self.client_lag_p50.set(metrics_snapshot['latency_p50'])
        if 'latency_p90' in metrics_snapshot:
            self.client_lag_p90.set(metrics_snapshot['latency_p90'])
        if 'latency_p99' in metrics_snapshot:
            self.client_lag_p99.set(metrics_snapshot['latency_p99'])

        # Update TTFT metrics
        if 'ttft_p50' in metrics_snapshot:
            self.ttft_p50.set(metrics_snapshot['ttft_p50'])
        if 'ttft_p90' in metrics_snapshot:
            self.ttft_p90.set(metrics_snapshot['ttft_p90'])
        if 'ttft_p99' in metrics_snapshot:
            self.ttft_p99.set(metrics_snapshot['ttft_p99'])

        # Update throughput
        if 'tps' in metrics_snapshot:
            self.tokens_per_second.set(metrics_snapshot['tps'])

        # Update request rate
        if 'rps' in metrics_snapshot:
            self.request_rate.set(metrics_snapshot['rps'])

        # Update error rates
        if 'error_rate' in metrics_snapshot:
            self.error_rate.set(metrics_snapshot['error_rate'])

        if 'contention_error_rate' in metrics_snapshot:
            self.contention_error_rate.set(metrics_snapshot['contention_error_rate'])

        # Update active users
        self.active_users.set(active_users)

    def increment_counters(
        self,
        requests_delta: int = 0,
        errors_delta: int = 0,
        tokens_delta: int = 0
    ):
        """
        Increments counter metrics.

        Args:
            requests_delta: Number of new requests since last update
            errors_delta: Number of new errors since last update
            tokens_delta: Number of new tokens generated since last update
        """
        if requests_delta > 0:
            self.total_requests.inc(requests_delta)
        if errors_delta > 0:
            self.total_errors.inc(errors_delta)
        if tokens_delta > 0:
            self.total_tokens.inc(tokens_delta)

    def push_metrics(self, timeout: int = 5) -> bool:
        """
        Pushes current metrics to Prometheus Pushgateway.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if push succeeded, False otherwise
        """
        try:
            # Prepare auth handler if credentials provided
            handler = None
            if self.username and self.password:
                def handler(url, method, timeout, headers, data):
                    return basic_auth_handler(
                        url, method, timeout, headers, data,
                        self.username, self.password
                    )

            # Push to gateway
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=self.job_name,
                registry=self.registry,
                timeout=timeout,
                handler=handler,
                grouping_key={'instance': self.instance_name}
            )

            logger.debug(f"Metrics pushed successfully to {self.pushgateway_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")
            return False

    def push_final_summary(self, summary_dict: Dict[str, float]):
        """
        Pushes final test summary as a one-time event.
        Useful for post-test analysis in Grafana.

        Args:
            summary_dict: Final test summary metrics
        """
        try:
            # Create temporary metrics for final summary
            final_registry = CollectorRegistry()

            for key, value in summary_dict.items():
                gauge = Gauge(
                    f'llm_final_{key}',
                    f'Final test metric: {key}',
                    registry=final_registry
                )
                gauge.set(value)

            # Push with different job name to separate from real-time metrics
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=f"{self.job_name}_final",
                registry=final_registry,
                grouping_key={'instance': self.instance_name}
            )

            logger.info("Final summary pushed to Pushgateway")

        except Exception as e:
            logger.error(f"Failed to push final summary: {e}")
