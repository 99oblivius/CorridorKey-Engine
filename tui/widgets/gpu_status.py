"""GPU status widget — compact utilization and VRAM display."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

from ..gpu_monitor import GPUMonitor, GPUStats


class GPUStatus(Static):
    """Compact GPU status: summed util% + highest VRAM with GPU ID."""

    DEFAULT_CSS = """
    GPUStatus {
        width: auto;
        height: 1;
        color: $text-muted;
    }

    GPUStatus.active {
        color: $text;
    }
    """

    utilization: reactive[int] = reactive(0)
    vram_used: reactive[float] = reactive(0.0)
    vram_total: reactive[float] = reactive(0.0)
    vram_gpu_id: reactive[int] = reactive(0)
    gpu_available: reactive[bool] = reactive(False)

    def __init__(self, **kwargs: object) -> None:
        super().__init__("GPU: N/A", **kwargs)
        self._monitor = GPUMonitor()

    def on_mount(self) -> None:
        if self._monitor.available:
            self.set_interval(1.0, self._poll_gpu)

    def _poll_gpu(self) -> None:
        stats = self._monitor.poll()
        self.utilization = stats.utilization_sum
        self.vram_used = stats.vram_used_gb
        self.vram_total = stats.vram_total_gb
        self.vram_gpu_id = stats.vram_gpu_id
        self.gpu_available = stats.available
        self._render_status()

    def update_from_stats(self, stats: GPUStats) -> None:
        """Update from externally provided stats."""
        self.utilization = stats.utilization_sum
        self.vram_used = stats.vram_used_gb
        self.vram_total = stats.vram_total_gb
        self.vram_gpu_id = stats.vram_gpu_id
        self.gpu_available = stats.available
        self._render_status()

    def _render_status(self) -> None:
        if not self.gpu_available:
            self.update("GPU: N/A")
            self.remove_class("active")
            return

        self.add_class("active")
        self.update(
            f"GPU: {self.utilization}% | VRAM: {self.vram_used:.1f}/{self.vram_total:.1f} GB (gpu{self.vram_gpu_id})"
        )
