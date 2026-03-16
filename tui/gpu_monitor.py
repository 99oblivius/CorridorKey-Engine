"""Non-blocking GPU stats polling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from textual.message import Message

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """Snapshot of GPU utilization and memory usage."""

    utilization_sum: int = 0  # Summed % across all GPUs
    vram_used_gb: float = 0.0  # Highest GPU's VRAM used (GB)
    vram_total_gb: float = 0.0  # Highest GPU's VRAM total (GB)
    vram_gpu_id: int = 0  # Which GPU has the highest usage
    available: bool = False


class GPUStatsUpdate(Message):
    """Posted when new GPU stats are available."""

    def __init__(self, stats: GPUStats) -> None:
        super().__init__()
        self.stats = stats


class GPUMonitor:
    """Polls GPU stats via pynvml.  Falls back gracefully if unavailable."""

    def __init__(self) -> None:
        self._available = False
        self._device_count = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._available = True
            self._device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def poll(self) -> GPUStats:
        """Return current GPU stats.  Non-blocking, fast syscalls."""
        if not self._available:
            return GPUStats()

        import pynvml

        util_sum = 0
        max_vram_used = 0.0
        max_vram_total = 0.0
        max_gpu_id = 0

        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_sum += util.gpu

                used_gb = mem.used / (1024**3)
                total_gb = mem.total / (1024**3)
                if used_gb > max_vram_used:
                    max_vram_used = used_gb
                    max_vram_total = total_gb
                    max_gpu_id = i
            except Exception:
                continue

        return GPUStats(
            utilization_sum=util_sum,
            vram_used_gb=max_vram_used,
            vram_total_gb=max_vram_total,
            vram_gpu_id=max_gpu_id,
            available=True,
        )
