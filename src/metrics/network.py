import asyncio
import logging
import time
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def _parse_ping_output(output: str) -> Optional[Dict[str, float]]:
    """
    Parse ping summary lines. Supports macOS/Linux output.
    """
    lines = output.splitlines()
    stats_line = next((l for l in lines if "min/avg" in l or "round-trip" in l), None)
    packet_line = next((l for l in lines if "packets transmitted" in l), None)

    if not stats_line:
        return None

    parts = stats_line.split("=")[-1].strip().split("/")
    if len(parts) < 3:
        return None

    min_ms, avg_ms, max_ms = parts[0:3]
    loss = None
    if packet_line and "%" in packet_line:
        try:
            loss = float(packet_line.split("%")[0].split()[-1])
        except Exception:
            loss = None

    return {
        "min_ms": float(min_ms),
        "avg_ms": float(avg_ms),
        "max_ms": float(max_ms),
        "loss_pct": loss
    }


class PingMonitor:
    def __init__(self, host: str, count: int = 4, interval_seconds: float = 60.0):
        self.host = host
        self.count = count
        self.interval_seconds = interval_seconds
        self.samples: List[Dict[str, float]] = []
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def run_once(self):
        cmd = ["ping", "-c", str(self.count), "-n", self.host]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"Ping to {self.host} failed: {err.decode().strip()}")
            return None

        parsed = _parse_ping_output(out.decode())
        if parsed:
            parsed["ts"] = time.time()
            self.samples.append(parsed)
        return parsed

    async def _loop(self):
        while not self._stop.is_set():
            await self.run_once()
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
