import unittest

from pydantic import ValidationError

from src.config.schema import GlobalConfig


BASE_WORKLOAD = {
    "users": 1,
    "duration_seconds": 1,
}

BASE_PROMPTS = {
    "min_tokens": 1,
    "max_tokens": 4,
}


class GlobalConfigValidationTests(unittest.TestCase):
    def test_requires_exactly_one_server_definition(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalConfig(
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
            )

        with self.assertRaises(ValidationError):
            GlobalConfig(
                server={"base_url": "http://localhost:8080/completion"},
                servers=[{"base_url": "http://localhost:8081/completion"}],
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
            )

    def test_rejects_non_positive_server_weights(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalConfig(
                servers=[
                    {"base_url": "http://localhost:8080/completion", "weight": 0},
                    {"base_url": "http://localhost:8081/completion", "weight": 1},
                ],
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
            )

    def test_comparison_mode_requires_multiple_servers(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalConfig(
                server={"base_url": "http://localhost:8080/completion"},
                comparison_mode=True,
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
            )

    def test_remote_system_source_requires_ssh_host(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalConfig(
                server={"base_url": "http://localhost:8080/completion"},
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
                system={"source": "ssh"},
            )

    def test_think_time_validation_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalConfig(
                server={"base_url": "http://localhost:8080/completion"},
                workload=BASE_WORKLOAD,
                prompts=BASE_PROMPTS,
                think_time={"enabled": True, "distribution": "uniform", "min_seconds": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
