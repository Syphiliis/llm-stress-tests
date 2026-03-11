import unittest

from src.metrics.stats import (
    ErrorCategory,
    RequestMetrics,
    StatsCalculator,
    categorize_error,
)


class StatsCalculatorTests(unittest.TestCase):
    def test_categorize_error_recognizes_expected_buckets(self) -> None:
        self.assertEqual(categorize_error("HTTP 503 Service Unavailable"), ErrorCategory.HTTP_5XX)
        self.assertEqual(categorize_error("Connection refused by peer"), ErrorCategory.CONNECTION_REFUSED)
        self.assertEqual(categorize_error("Request timed out"), ErrorCategory.TIMEOUT)
        self.assertEqual(categorize_error("unexpected failure"), ErrorCategory.OTHER)

    def test_calculate_summary_marks_complete_failure_as_na(self) -> None:
        calc = StatsCalculator()
        calc.start_test()
        calc.start_time = 100.0
        calc.end_time = 110.0
        calc.add_metric(
            RequestMetrics(
                request_id="req-1",
                start_time=100.0,
                end_time=101.0,
                error="timeout while waiting for first token",
            )
        )

        summary = calc.calculate_summary()

        self.assertTrue(summary.is_failed)
        self.assertEqual(summary.total_requests, 1)
        self.assertEqual(summary.successful_requests, 0)
        self.assertEqual(summary.failed_requests, 1)
        self.assertIsNone(summary.latency_p50)
        self.assertIsNone(summary.ttft_p50)
        self.assertEqual(summary.error_breakdown["timeout"], 1)

    def test_calculate_summary_tracks_dynamic_load_fields(self) -> None:
        calc = StatsCalculator()
        calc.start_test()
        calc.start_time = 100.0
        calc.end_time = 110.0
        calc.add_metric(
            RequestMetrics(
                request_id="req-1",
                start_time=100.0,
                end_time=101.0,
                ttft=0.2,
                input_tokens=256,
                output_tokens=512,
                queue_wait=0.1,
                concurrency=3,
                user_id=1,
                target_users=10,
                target_rps=4.0,
                in_flight=3,
                think_time=1.0,
            )
        )
        calc.add_metric(
            RequestMetrics(
                request_id="req-2",
                start_time=102.0,
                end_time=104.0,
                ttft=0.4,
                input_tokens=512,
                output_tokens=256,
                queue_wait=0.3,
                concurrency=5,
                user_id=2,
                target_users=14,
                target_rps=8.0,
                in_flight=5,
                think_time=2.0,
            )
        )

        summary = calc.calculate_summary()

        self.assertEqual(summary.total_requests, 2)
        self.assertAlmostEqual(summary.target_users_mean, 12.0)
        self.assertAlmostEqual(summary.target_rps_mean, 6.0)
        self.assertEqual(summary.max_in_flight, 5)
        self.assertAlmostEqual(summary.think_time_mean, 1.5)
        self.assertIsNotNone(summary.queue_wait_p50)
        self.assertIsNotNone(summary.tokens_vs_concurrency)
        self.assertIn("3", summary.tokens_vs_concurrency)
        self.assertIn("5", summary.tokens_vs_concurrency)

    def test_get_current_snapshot_returns_na_before_any_success(self) -> None:
        calc = StatsCalculator()
        calc.start_test()

        snapshot = calc.get_current_snapshot()

        self.assertIsNone(snapshot["latency_p50"])
        self.assertIsNone(snapshot["ttft_p50"])
        self.assertEqual(snapshot["rps"], 0.0)
        self.assertEqual(snapshot["tps"], 0.0)


if __name__ == "__main__":
    unittest.main()
