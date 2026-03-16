"""TUI engine client -- bridges EngineClient events to Textual messages.

This module replaces ``pipeline_bridge.py``.  Instead of attaching Python
log handlers and running pipeline code in-process, it spawns an engine
subprocess and converts its JSON-RPC events into Textual ``Message``
objects posted to the event loop.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

from textual.message import Message

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Textual Messages posted from engine events
# ---------------------------------------------------------------------------


class EngineClipStarted(Message):
    """A clip has started processing."""

    def __init__(
        self,
        clip: str,
        frames: int,
        clip_index: int = 0,
        clips_total: int = 1,
    ) -> None:
        super().__init__()
        self.clip = clip
        self.frames = frames
        self.clip_index = clip_index
        self.clips_total = clips_total


class EngineProgress(Message):
    """Frame progress update."""

    def __init__(
        self,
        clip: str,
        done: int,
        total: int,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> None:
        super().__init__()
        self.clip = clip
        self.done = done
        self.total = total
        self.bytes_read = bytes_read
        self.bytes_written = bytes_written


class EngineJobCompleted(Message):
    """Job completed successfully."""

    def __init__(
        self,
        clips_ok: int = 0,
        clips_failed: int = 0,
        total_frames: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self.clips_ok = clips_ok
        self.clips_failed = clips_failed
        self.total_frames = total_frames
        self.elapsed_seconds = elapsed_seconds


class EngineJobFailed(Message):
    """Job failed."""

    def __init__(self, error: str = "") -> None:
        super().__init__()
        self.error = error


class EngineJobCancelled(Message):
    """Job was cancelled."""

    def __init__(self, frames_completed: int = 0) -> None:
        super().__init__()
        self.frames_completed = frames_completed


class EngineLogMessage(Message):
    """A log message from the engine."""

    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__()
        self.text = text
        self.level = level


class EngineModelLoading(Message):
    """Model is being loaded."""

    def __init__(self, model: str = "", device: str = "") -> None:
        super().__init__()
        self.model = model
        self.device = device


class EngineModelLoaded(Message):
    """Model finished loading."""

    def __init__(
        self,
        model: str = "",
        device: str = "",
        vram_mb: float = 0.0,
        load_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.vram_mb = vram_mb
        self.load_seconds = load_seconds


# ---------------------------------------------------------------------------
# TUI Engine Client
# ---------------------------------------------------------------------------


class TUIEngineClient:
    """Wraps EngineClient for use in the Textual TUI.

    Spawns an engine subprocess and runs a background thread that reads
    events and posts them as Textual Messages to a target widget.
    """

    def __init__(self, app: App, target: Widget | None = None) -> None:
        from ck_engine.client import EngineClient

        self._app = app
        self._target = target  # If None, events aren't posted to any widget
        self._client = EngineClient.spawn()
        self._active_job_id: str | None = None
        self._event_thread: threading.Thread | None = None
        self._stopped = False

    def _post(self, message: Message) -> None:
        """Post a Textual message to the target widget via the event loop."""
        if self._target is None:
            return
        try:
            loop = self._app._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._target.post_message, message)
        except Exception:
            pass

    def set_target(self, target: Widget | None) -> None:
        """Change the widget that receives event messages."""
        self._target = target

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None

    def capabilities(self) -> dict:
        return self._client.capabilities()

    def scan_project(self, path: str) -> dict:
        return self._client.scan_project(path)

    def submit_generate(self, params: Any) -> str:
        """Submit a generate job and start listening for events."""
        job_id = self._client.submit_generate(params)
        self._active_job_id = job_id
        self._start_event_listener()
        return job_id

    def submit_inference(self, params: Any) -> str:
        """Submit an inference job and start listening for events."""
        job_id = self._client.submit_inference(params)
        self._active_job_id = job_id
        self._start_event_listener()
        return job_id

    def cancel_job(self) -> None:
        """Cancel the active job."""
        if self._active_job_id:
            try:
                self._client.cancel_job(self._active_job_id)
            except Exception:
                pass

    def subscribe(self, categories: list[str]) -> None:
        self._client.subscribe(categories)

    def unsubscribe(self, categories: list[str]) -> None:
        self._client.unsubscribe(categories)

    def model_status(self) -> dict:
        return self._client.model_status()

    def unload_models(self, which: str = "all") -> float:
        return self._client.unload_models(which)

    def _start_event_listener(self) -> None:
        """Start a background thread that reads events and posts Textual messages."""
        if self._event_thread is not None and self._event_thread.is_alive():
            return  # Already listening

        self._stopped = False
        self._event_thread = threading.Thread(
            target=self._event_loop, name="tui-engine-events", daemon=True
        )
        self._event_thread.start()

    def _event_loop(self) -> None:
        """Background thread: read events from engine, post as Textual messages."""
        from ck_engine.api.events import (
            ClipStarted,
            JobCancelled,
            JobCompleted,
            JobFailed,
            JobProgress,
            LogEvent,
            ModelLoaded,
            ModelLoading,
        )

        try:
            for event in self._client.iter_events(timeout=600.0):
                if self._stopped:
                    break

                # Convert engine event to Textual message
                if isinstance(event, ClipStarted):
                    self._post(EngineClipStarted(
                        clip=event.clip,
                        frames=event.frames,
                        clip_index=event.clip_index,
                        clips_total=event.clips_total,
                    ))

                elif isinstance(event, JobProgress):
                    self._post(EngineProgress(
                        clip=event.clip,
                        done=event.done,
                        total=event.total,
                        bytes_read=event.bytes_read,
                        bytes_written=event.bytes_written,
                    ))

                elif isinstance(event, JobCompleted):
                    self._post(EngineJobCompleted(
                        clips_ok=event.clips_ok,
                        clips_failed=event.clips_failed,
                        total_frames=event.total_frames,
                        elapsed_seconds=event.elapsed_seconds,
                    ))
                    self._active_job_id = None
                    break

                elif isinstance(event, JobFailed):
                    self._post(EngineJobFailed(error=event.error))
                    self._active_job_id = None
                    break

                elif isinstance(event, JobCancelled):
                    self._post(EngineJobCancelled(
                        frames_completed=event.frames_completed,
                    ))
                    self._active_job_id = None
                    break

                elif isinstance(event, ModelLoading):
                    self._post(EngineModelLoading(
                        model=event.model,
                        device=event.device,
                    ))

                elif isinstance(event, ModelLoaded):
                    self._post(EngineModelLoaded(
                        model=event.model,
                        device=event.device,
                        vram_mb=event.vram_mb,
                        load_seconds=event.load_seconds,
                    ))

                elif isinstance(event, LogEvent):
                    self._post(EngineLogMessage(
                        text=event.message,
                        level=event.level,
                    ))

        except queue.Empty:
            pass  # Timeout -- job may be very long
        except Exception as exc:
            logger.debug("Event listener error: %s", exc)
            self._active_job_id = None

    def close(self, shutdown: bool = True) -> None:
        """Close the engine connection."""
        self._stopped = True
        try:
            self._client.close(shutdown=shutdown)
        except Exception:
            pass
        if self._event_thread is not None:
            self._event_thread.join(timeout=3.0)
            self._event_thread = None
