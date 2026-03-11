import unittest
from unittest.mock import patch

from src.metrics.system_sampler import SystemSampler


class SystemSamplerTests(unittest.TestCase):
    def test_builds_local_command(self) -> None:
        sampler = SystemSampler(gpu_command="python3 --version")

        with patch("src.metrics.system_sampler.shutil.which", return_value="/usr/bin/python3"):
            self.assertEqual(sampler._build_gpu_command(), ["python3", "--version"])

    def test_builds_ssh_command(self) -> None:
        sampler = SystemSampler(
            gpu_command="nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
            source="ssh",
            ssh_host="gpu.example.com",
            ssh_user="ubuntu",
            ssh_port=2222,
        )

        with patch("src.metrics.system_sampler.shutil.which", return_value="/usr/bin/ssh"):
            self.assertEqual(
                sampler._build_gpu_command(),
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=5",
                    "-p",
                    "2222",
                    "ubuntu@gpu.example.com",
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
                ],
            )

    def test_collect_gpu_parses_three_column_output(self) -> None:
        sampler = SystemSampler(gpu_command="nvidia-smi")

        with patch.object(sampler, "_build_gpu_command", return_value=["nvidia-smi"]):
            with patch("src.metrics.system_sampler.subprocess.check_output", return_value="95, 1024, 16384\n"):
                self.assertEqual(
                    sampler._collect_gpu(),
                    [
                        {
                            "utilization_gpu": 95.0,
                            "memory_used_mb": 1024.0,
                            "memory_total_mb": 16384.0,
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
