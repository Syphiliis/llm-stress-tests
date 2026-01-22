import asyncio
import logging
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
    Periodically collects CPU per-core, RAM, and GPU utilization (if nvidia-smi available).
    """

    def __init__(self, interval_seconds: float = 15.0, gpu_command: Optional[str] = None):
        self.interval_seconds = interval_seconds
        self.gpu_command = gpu_command
        self.snapshots: List[Dict] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def _collect_snapshot(self):
        snapshot: Dict = {"ts": time.time()}

        if psutil:
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
        if not shutil.which(self.gpu_command.split()[0]):
            return None
        try:
            output = subprocess.check_output(self.gpu_command, shell=True, text=True).strip().splitlines()
            results = []
            for line in output:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    results.append({
                        "utilization_gpu": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    })
            return results or None
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
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
