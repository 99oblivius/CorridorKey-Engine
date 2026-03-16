"""Inference panel — CorridorKey inference with settings and progress."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, RichLog, Static

from ..client import (
    EngineClipStarted,
    EngineJobCancelled,
    EngineJobCompleted,
    EngineJobFailed,
    EngineLogMessage,
    EngineModelLoaded,
    EngineModelLoading,
    EngineProgress,
)
from ..widgets.progress_panel import ProgressPanel

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ck_engine.api.types import InferenceSettings as APISettings

logger = logging.getLogger(__name__)


def _fmt_io_speed(bps: float) -> str:
    """Format bytes/second as human-readable throughput."""
    if bps < 1024:
        return f"{bps:.0f} B/s"
    if bps < 1024 * 1024:
        return f"{bps / 1024:.1f} KB/s"
    if bps < 1024 * 1024 * 1024:
        return f"{bps / (1024 * 1024):.1f} MB/s"
    return f"{bps / (1024 * 1024 * 1024):.2f} GB/s"


class _IOSpeedTracker:
    """3-second exponentially-weighted moving average for IO throughput."""

    def __init__(self, window: float = 3.0) -> None:
        self._window = window
        self._samples_r: list[tuple[float, int]] = []
        self._samples_w: list[tuple[float, int]] = []

    def record(self, now: float, total_read: int, total_written: int) -> None:
        self._samples_r.append((now, total_read))
        self._samples_w.append((now, total_written))
        cutoff = now - self._window * 2
        self._samples_r = [(t, b) for t, b in self._samples_r if t >= cutoff]
        self._samples_w = [(t, b) for t, b in self._samples_w if t >= cutoff]

    def speeds(self) -> tuple[float, float]:
        return (self._ewma(self._samples_r), self._ewma(self._samples_w))

    def _ewma(self, samples: list[tuple[float, int]]) -> float:
        if len(samples) < 2:
            return 0.0
        now = samples[-1][0]
        tau = self._window
        weighted_rate = 0.0
        total_weight = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i - 1][0]
            if dt <= 0:
                continue
            rate = (samples[i][1] - samples[i - 1][1]) / dt
            age = now - samples[i][0]
            weight = 2.0 ** (-age / tau)
            weighted_rate += rate * weight
            total_weight += weight
        return weighted_rate / total_weight if total_weight > 0 else 0.0

    def reset(self) -> None:
        self._samples_r.clear()
        self._samples_w.clear()


class InferencePanel(Vertical, can_focus=True):
    """Inference execution panel with settings and progress monitoring."""

    DEFAULT_CSS = """
    InferencePanel {
        layout: vertical;
    }

    InferencePanel #inf-log {
        height: 1fr;
        min-height: 0;
        border: solid $border;
        margin: 0 1;
    }

    InferencePanel #inf-bottom {
        height: auto;
    }

    InferencePanel #inf-settings-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    InferencePanel #inf-progress {
        margin: 0 1;
    }

    InferencePanel #inf-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    InferencePanel #inf-button-bar {
        height: auto;
        min-height: 3;
        padding: 1 1;
        align: left middle;
    }

    InferencePanel #inf-button-bar Button {
        min-width: 16;
        margin-right: 2;
    }

    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "start_inference", "Start", priority=True),
        Binding("escape", "cancel_inference", "Cancel", priority=True),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._job_id: str | None = None
        self._current_clip: str = ""
        self._io_tracker = _IOSpeedTracker(window=3.0)

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, id="inf-log")
        with Vertical(id="inf-bottom"):
            yield self._build_settings_bar()
            yield ProgressPanel(id="inf-progress")
            yield Static("", id="inf-status")
            with Horizontal(id="inf-button-bar"):
                yield Button("Start", variant="primary", id="inf-start-btn")
                yield Button("Cancel", variant="default", id="inf-cancel-btn")

    def _build_settings_bar(self) -> Static:
        """Build the settings summary bar from current project settings."""
        settings = self._load_settings()
        cs = "Linear" if settings.input_is_linear else "sRGB"
        despill = int(settings.despill_strength * 10)
        despeckle = f"ON ({settings.despeckle_size})" if settings.auto_despeckle else "OFF"
        text = f"Settings: {cs} | despill {despill} | despeckle {despeckle} | refiner {settings.refiner_scale}"
        return Static(text, id="inf-settings-bar")

    def on_mount(self) -> None:
        self.query_one("#inf-log", RichLog).can_focus = False

    # ------------------------------------------------------------------
    # Engine event handlers
    # ------------------------------------------------------------------

    @on(EngineLogMessage)
    def _on_log_message(self, event: EngineLogMessage) -> None:
        log = self.query_one("#inf-log", RichLog)
        if event.level == "error":
            log.write(f"[red]ERROR:[/red] {event.text}")
        elif event.level == "warning":
            log.write(f"[yellow]{event.text}[/yellow]")
        else:
            log.write(event.text)

    @on(EngineClipStarted)
    def _on_clip_started(self, event: EngineClipStarted) -> None:
        self._current_clip = event.clip
        self._io_tracker.reset()
        panel = self.query_one("#inf-progress", ProgressPanel)
        panel.activate_clip(event.clip, event.frames)
        self._set_status(f"Processing: {event.clip}")

    @on(EngineProgress)
    def _on_progress(self, event: EngineProgress) -> None:
        if self._current_clip:
            import time as _time

            panel = self.query_one("#inf-progress", ProgressPanel)
            self._io_tracker.record(
                _time.monotonic(), event.bytes_read, event.bytes_written,
            )
            r_bps, w_bps = self._io_tracker.speeds()
            r_speed = _fmt_io_speed(r_bps) if r_bps > 0 else ""
            w_speed = _fmt_io_speed(w_bps) if w_bps > 0 else ""
            panel.update_progress(
                self._current_clip, event.done, event.total,
                read_speed=r_speed, write_speed=w_speed,
            )

    @on(EngineJobCompleted)
    def _on_job_completed(self, event: EngineJobCompleted) -> None:
        self._set_status("Inference complete.")
        self._finish_job()

    @on(EngineJobFailed)
    def _on_job_failed(self, event: EngineJobFailed) -> None:
        self._set_status(f"Error: {event.error}")
        self.query_one("#inf-log", RichLog).write(f"[red]ERROR:[/red] {event.error}")
        self._finish_job()

    @on(EngineJobCancelled)
    def _on_job_cancelled(self, event: EngineJobCancelled) -> None:
        self._set_status("Inference cancelled.")
        self._finish_job()

    @on(EngineModelLoading)
    def _on_model_loading(self, event: EngineModelLoading) -> None:
        self.query_one("#inf-log", RichLog).write(f"Loading model {event.model} on {event.device}...")

    @on(EngineModelLoaded)
    def _on_model_loaded(self, event: EngineModelLoaded) -> None:
        self.query_one("#inf-log", RichLog).write(
            f"Model loaded in {event.load_seconds:.1f}s ({event.vram_mb:.0f} MB VRAM)"
        )

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def _finish_job(self) -> None:
        self._job_id = None
        self.query_one("#inf-start-btn", Button).disabled = False

    @on(Button.Pressed, "#inf-start-btn")
    def _on_start_btn(self, event: Button.Pressed) -> None:
        self.action_start_inference()

    @on(Button.Pressed, "#inf-cancel-btn")
    def _on_cancel_btn(self, event: Button.Pressed) -> None:
        self.action_cancel_inference()

    def action_start_inference(self) -> None:
        """Submit inference job to the engine."""
        if self._job_id is not None:
            return

        path = self._get_project_path()
        if not path:
            self._set_status("No project loaded. Go to Clips tab first.")
            return

        engine = self.app.engine
        if engine is None:
            self._set_status("Engine not available.")
            return
        engine.set_target(self)

        # Build API settings from project settings
        from ck_engine.api.types import InferenceParams
        from ck_engine.api.types import InferenceSettings as APISetts
        from ck_engine.api.types import OptimizationParams

        pipeline_settings = self._load_settings()
        settings = APISetts(
            input_is_linear=pipeline_settings.input_is_linear,
            despill_strength=pipeline_settings.despill_strength,
            auto_despeckle=pipeline_settings.auto_despeckle,
            despeckle_size=pipeline_settings.despeckle_size,
            refiner_scale=pipeline_settings.refiner_scale,
        )

        # Build optimization from global settings
        opt = self._build_optimization_params()

        # Build device list from global settings
        from ck_engine.settings import GlobalSettings

        gs = GlobalSettings.load()
        devices = None
        if gs.devices:
            device_type = gs.device if gs.device not in ("auto", "") else "cuda"
            devices = [f"{device_type}:{idx.strip()}" for idx in gs.devices]

        params = InferenceParams(
            path=path,
            device=gs.device if gs.device != "auto" else "auto",
            backend=gs.backend if gs.backend != "auto" else "auto",
            settings=settings,
            optimization=opt,
            devices=devices,
            img_size=gs.img_size,
            read_workers=gs.read_workers,
            write_workers=gs.write_workers,
            cpus=gs.cpus,
            gpu_resilience=gs.gpu_resilience,
        )

        try:
            self._job_id = engine.submit_inference(params)
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            return

        self.query_one("#inf-start-btn", Button).disabled = True
        self.query_one("#inf-progress", ProgressPanel).clear()
        self._register_recent_project()
        self._set_status("Starting inference...")

    def action_cancel_inference(self) -> None:
        """Cancel the running inference job."""
        if self._job_id is not None:
            engine = self.app.engine
            if engine:
                engine.cancel_job()
            self._set_status("Cancelling...")
        else:
            self.app.action_goto("clip_manager")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_optimization_params():
        """Build API OptimizationParams from GlobalSettings."""
        from ck_engine.api.types import OptimizationParams
        from ck_engine.settings import GlobalSettings

        gs = GlobalSettings.load()
        return OptimizationParams(
            flash_attention=gs.flash_attention or None,
            tiled_refiner=gs.tiled_refiner or None,
            tile_size=gs.tile_size,
            tile_overlap=gs.tile_overlap,
            cache_clearing=gs.cache_clearing or None,
            disable_cudnn_benchmark=gs.disable_cudnn_benchmark or None,
            compile_mode=gs.compile_mode or None,
            model_precision=gs.precision,
            gpu_postprocess=gs.gpu_postprocess,
            comp_format=gs.comp_format,
            comp_checkerboard=gs.comp_checkerboard,
            dma_buffers=gs.dma_buffers,
        )

    def _load_settings(self) -> APISettings:
        """Load inference settings from project settings or defaults."""
        from ck_engine.api.types import InferenceSettings as APISetts

        try:
            from ck_engine.settings import ProjectSettings

            path = self._get_project_path()
            if path:
                from pathlib import Path

                ps = ProjectSettings.load(Path(path))
                return APISetts(
                    input_is_linear=ps.input_is_linear,
                    despill_strength=ps.despill_strength,
                    auto_despeckle=ps.auto_despeckle,
                    despeckle_size=ps.despeckle_size,
                    refiner_scale=ps.refiner_scale,
                )
        except Exception:
            pass
        return APISetts()

    def _get_project_path(self) -> str | None:
        """Get the current project path from the clip manager's path input."""
        try:
            from textual.widgets import Input
            inp = self.app.query_one("#path-input", Input)
            path = inp.value.strip()
            return path if path and os.path.isdir(path) else None
        except Exception:
            return None

    def check_readiness(self) -> None:
        """Disable Start button and show status when no clips are ready."""
        path = self._get_project_path()
        start_btn = self.query_one("#inf-start-btn", Button)
        if not path:
            start_btn.disabled = True
            self._set_status("No project loaded. Load a project in the Clips tab.")
            return
        try:
            engine = self.app.engine
            if engine is None:
                start_btn.disabled = True
                return
            result = engine.scan_project(path)
            clips = result.get("clips", [])
            ready = [c for c in clips
                     if c.get("input") and c["input"]["frame_count"] > 0
                     and c.get("alpha") and c["alpha"]["frame_count"] > 0]
            if ready:
                start_btn.disabled = False
                self._set_status(f"{len(ready)} clip(s) ready for inference.")
            else:
                start_btn.disabled = True
                missing = [c["name"] for c in clips
                           if c.get("input") and c["input"]["frame_count"] > 0
                           and (not c.get("alpha") or c["alpha"]["frame_count"] == 0)]
                if missing:
                    self._set_status(f"Missing mattes: {', '.join(missing)}. Generate mattes first.")
                else:
                    self._set_status("No clips with input frames found.")
        except Exception:
            start_btn.disabled = True
            self._set_status("Could not scan project.")

    def _set_status(self, text: str) -> None:
        """Update the status bar text."""
        self.query_one("#inf-status", Static).update(text)

    def _register_recent_project(self) -> None:
        """Add the current project path to recent projects list."""
        try:
            from textual.widgets import Input

            from tui.screens.clip_manager import ClipManagerPanel
            from ck_engine.settings import GlobalSettings

            inp = self.app.query_one("#path-input", Input)
            path = inp.value.strip()
            if path and os.path.isdir(path):
                settings = GlobalSettings.load()
                settings.add_recent_project(path)
                settings.save()
                # Refresh the clip manager's recent list
                panel = self.app.query_one(ClipManagerPanel)
                panel.refresh_recent()
        except Exception:
            pass
