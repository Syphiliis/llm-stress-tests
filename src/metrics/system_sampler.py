import asyncio
import logging
import shlex
import shutil
import subprocess
import time
from typing import List, Dict, Optional

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # Optional dependency

logger = logging.getLogger(__name__)


class SystemSampler:
    """
    Periodically collects CPU/RAM locally and GPU utilization locally or over SSH.
    """

    def __init__(
        self,
        interval_seconds: float = 15.0,
        gpu_command: Optional[str] = None,
        source: str = "local",
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_port: int = 22,
        ssh_options: Optional[List[str]] = None,
        include_local_cpu_ram: bool = False,
    ):
        self.interval_seconds = interval_seconds
        self.gpu_command = gpu_command
        self.source = source
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.ssh_options = ssh_options or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
        self.include_local_cpu_ram = include_local_cpu_ram
        self.snapshots: List[Dict] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def _collect_snapshot(self):
        snapshot: Dict = {"ts": time.time(), "system_source": self.source}

        should_collect_local_host = self.source == "local" or self.include_local_cpu_ram
        if should_collect_local_host and psutil:
            snapshot["cpu_per_core"] = psutil.cpu_percent(percpu=True)
            snapshot["cpu_avg"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            snapshot["ram_used_mb"] = mem.used / (1024 * 1024)
            snapshot["ram_percent"] = mem.percent
        else:
            snapshot["cpu_per_core"] = None
            snapshot["cpu_avg"] = None
            snapshot["ram_used_mb"] = None
            snapshot["ram_percent"] = None

        gpu_metrics = self._collect_gpu()
        if gpu_metrics:
            snapshot["gpu"] = gpu_metrics

        self.snapshots.append(snapshot)

    def _collect_gpu(self) -> Optional[List[Dict[str, float]]]:
        if not self.gpu_command:
            return None

        command = self._build_gpu_command()
        if not command:
            return None

        try:
            output = subprocess.check_output(command, text=True, timeout=15).strip().splitlines()
            results = []
            for line in output:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    row = {
                        "utilization_gpu": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    }
                    if len(parts) >= 4:
                        row["power_draw_watts"] = float(parts[3])
                    results.append(row)
            return results or None
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
            return None

    def _build_gpu_command(self) -> Optional[List[str]]:
        if self.source == "local":
            command = shlex.split(self.gpu_command)
            if not command or not shutil.which(command[0]):
                return None
            return command

        if self.source == "ssh":
            if not self.ssh_host or not shutil.which("ssh"):
                return None
            target = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
            return ["ssh", *self.ssh_options, "-p", str(self.ssh_port), target, self.gpu_command]

        return None

    async def _loop(self):
        while not self._stop.is_set():
            await self._collect_snapshot()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task
