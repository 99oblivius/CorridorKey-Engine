"""Generate Mattes panel — alpha generation with BiRefNet/GVM/VideoMaMa."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, RichLog, Select, Static

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

_MODELS = [
    ("BiRefNet  (~4 GB VRAM, per-frame)", "birefnet"),
    ("GVM  (temporal, higher VRAM)", "gvm"),
    ("VideoMaMa  (temporal, requires mask)", "videomama"),
]


class GeneratePanel(Vertical, can_focus=True):
    """Alpha matte generation panel."""

    DEFAULT_CSS = """
    GeneratePanel {
        layout: vertical;
    }

    GeneratePanel #gen-log {
        height: 1fr;
        min-height: 0;
        border: solid $border;
        margin: 0 1;
    }

    GeneratePanel #gen-bottom {
        height: auto;
    }

    GeneratePanel #model-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    GeneratePanel #model-label {
        width: 8;
        content-align: left middle;
        text-style: bold;
    }

    GeneratePanel #model-select {
        width: 50;
    }

    GeneratePanel #mode-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }

    GeneratePanel #mode-label {
        width: 8;
        content-align: left middle;
        text-style: bold;
    }

    GeneratePanel #mode-select {
        width: 50;
    }

    GeneratePanel #gen-progress {
        margin: 0 1;
    }

    GeneratePanel #gen-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    GeneratePanel #gen-button-bar {
        height: auto;
        min-height: 3;
        padding: 1 1;
        align: left middle;
    }

    GeneratePanel #gen-button-bar Button {
        min-width: 16;
        margin-right: 2;
    }

    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "start_generation", "Start", priority=True),
        Binding("escape", "cancel_generation", "Cancel", priority=True),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._job_id: str | None = None
        self._current_clip: str = ""

    def compose(self) -> ComposeResult:
        # Log takes most space
        yield RichLog(highlight=True, markup=True, id="gen-log")
        # Bottom section: controls, progress, status, buttons
        with Vertical(id="gen-bottom"):
            with Horizontal(id="model-row"):
                yield Static("Model:", id="model-label")
                yield Select(
                    _MODELS,
                    value="birefnet",
                    id="model-select",
                )
            with Horizontal(id="mode-row"):
                yield Static("Mode:", id="mode-label")
                yield Select(
                    [
                        ("Replace (regenerate all)", "replace"),
                        ("Fill (only missing frames)", "fill"),
                        ("Skip (do nothing if exists)", "skip"),
                    ],
                    value="replace",
                    id="mode-select",
                )
            yield ProgressPanel(id="gen-progress")
            yield Static("", id="gen-status")
            with Horizontal(id="gen-button-bar"):
                yield Button("Start", variant="primary", id="gen-start-btn")
                yield Button("Cancel", variant="default", id="gen-cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#gen-log", RichLog).can_focus = False

    # ------------------------------------------------------------------
    # Engine event handlers
    # ------------------------------------------------------------------

    @on(EngineLogMessage)
    def _on_log_message(self, event: EngineLogMessage) -> None:
        log = self.query_one("#gen-log", RichLog)
        if event.level == "error":
            log.write(f"[red]ERROR:[/red] {event.text}")
        elif event.level == "warning":
            log.write(f"[yellow]{event.text}[/yellow]")
        else:
            log.write(event.text)

    @on(EngineClipStarted)
    def _on_clip_started(self, event: EngineClipStarted) -> None:
        self._current_clip = event.clip
        panel = self.query_one("#gen-progress", ProgressPanel)
        panel.activate_clip(event.clip, event.frames)
        self._set_status(f"Generating alpha for: {event.clip}")

    @on(EngineProgress)
    def _on_progress(self, event: EngineProgress) -> None:
        if self._current_clip:
            panel = self.query_one("#gen-progress", ProgressPanel)
            panel.update_progress(self._current_clip, event.done, event.total)

    @on(EngineJobCompleted)
    def _on_job_completed(self, event: EngineJobCompleted) -> None:
        self._set_status(f"Generation complete — {event.clips_ok} clip(s) processed.")
        self._finish_job()

    @on(EngineJobFailed)
    def _on_job_failed(self, event: EngineJobFailed) -> None:
        self._set_status(f"Error: {event.error}")
        self.query_one("#gen-log", RichLog).write(f"[red]ERROR:[/red] {event.error}")
        self._finish_job()

    @on(EngineJobCancelled)
    def _on_job_cancelled(self, event: EngineJobCancelled) -> None:
        self._set_status("Generation cancelled.")
        self._finish_job()

    @on(EngineModelLoading)
    def _on_model_loading(self, event: EngineModelLoading) -> None:
        self.query_one("#gen-log", RichLog).write(
            f"Loading model {event.model} on {event.device}..."
        )

    @on(EngineModelLoaded)
    def _on_model_loaded(self, event: EngineModelLoaded) -> None:
        self.query_one("#gen-log", RichLog).write(
            f"Model loaded in {event.load_seconds:.1f}s ({event.vram_mb:.0f} MB VRAM)"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _finish_job(self) -> None:
        self._job_id = None
        self.query_one("#gen-start-btn", Button).disabled = False

    @on(Button.Pressed, "#gen-start-btn")
    def _on_start_btn(self, event: Button.Pressed) -> None:
        self.action_start_generation()

    @on(Button.Pressed, "#gen-cancel-btn")
    def _on_cancel_btn(self, event: Button.Pressed) -> None:
        self.action_cancel_generation()

    def action_start_generation(self) -> None:
        """Submit alpha generation to the engine."""
        if self._job_id is not None:
            return

        select = self.query_one("#model-select", Select)
        model = str(select.value) if select.value != Select.BLANK else "birefnet"

        mode_select = self.query_one("#mode-select", Select)
        mode_str = str(mode_select.value) if mode_select.value != Select.BLANK else "replace"

        path = self._get_project_path()
        if not path:
            self._set_status("No project loaded. Go to Clips tab first.")
            return

        # Set the engine's event target to this panel
        engine = self.app.engine
        if engine is None:
            self._set_status("Engine not available.")
            return
        engine.set_target(self)

        from ck_engine.api.types import GenerateParams

        params = GenerateParams(path=path, model=model, mode=mode_str)

        try:
            self._job_id = engine.submit_generate(params)
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            return

        self.query_one("#gen-start-btn", Button).disabled = True
        self.query_one("#gen-progress", ProgressPanel).clear()
        self._set_status(f"Starting {model} generation ({mode_str} mode)...")

    def _set_status(self, text: str) -> None:
        """Update the status bar text."""
        self.query_one("#gen-status", Static).update(text)

    def action_cancel_generation(self) -> None:
        """Cancel the running generation."""
        if self._job_id is not None:
            engine = self.app.engine
            if engine:
                engine.cancel_job()
            self._set_status("Cancelling...")
        else:
            self.app.action_goto("clip_manager")

    # ------------------------------------------------------------------
    # Readiness check
    # ------------------------------------------------------------------

    def check_readiness(self) -> None:
        """Disable Start button and show status when no clips are ready."""
        path = self._get_project_path()
        start_btn = self.query_one("#gen-start-btn", Button)
        if not path:
            start_btn.disabled = True
            self._set_status("No project loaded. Load a project in the Clips tab.")
            return
        try:
            engine = self.app.engine
            if engine is None:
                start_btn.disabled = True
                self._set_status("Engine not available.")
                return
            result = engine.scan_project(path)
            clips = result.get("clips", [])
            valid = [c for c in clips if c.get("input") and c["input"]["frame_count"] > 0]
            if valid:
                start_btn.disabled = False
                self._set_status(f"{len(valid)} clip(s) ready for generation.")
            else:
                start_btn.disabled = True
                self._set_status("No clips with input frames found.")
        except Exception:
            start_btn.disabled = True
            self._set_status("Could not scan project.")

    def _get_project_path(self) -> str | None:
        """Retrieve the project path from the Clip Manager's path input."""
        try:
            inp = self.app.query_one("#path-input", Input)
            path = inp.value.strip()
            return path if path and os.path.isdir(path) else None
        except Exception:
            return None
