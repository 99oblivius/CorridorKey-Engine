"""Thread-safe bridge between backend pipeline callbacks and Textual messages."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from textual.message import Message

if TYPE_CHECKING:
    from textual.app import App


# ---------------------------------------------------------------------------
# Messages posted to the Textual event loop
# ---------------------------------------------------------------------------


class ClipStarted(Message):
    """A new clip has begun processing."""

    def __init__(self, clip_name: str, total_frames: int) -> None:
        super().__init__()
        self.clip_name = clip_name
        self.total_frames = total_frames


class ProgressUpdate(Message):
    """Frame progress update for the current clip."""

    def __init__(
        self,
        done: int,
        total: int,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> None:
        super().__init__()
        self.done = done
        self.total = total
        self.bytes_read = bytes_read
        self.bytes_written = bytes_written


class GenerationComplete(Message):
    """Alpha generation finished (success or cancellation)."""

    def __init__(self, succeeded: int = 0) -> None:
        super().__init__()
        self.succeeded = succeeded


class InferenceComplete(Message):
    """Inference run finished."""


class PipelineError(Message):
    """An error occurred during pipeline execution."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class LogMessage(Message):
    """A log record to display in the log pane."""

    def __init__(self, text: str, level: int = logging.INFO) -> None:
        super().__init__()
        self.text = text
        self.level = level


# ---------------------------------------------------------------------------
# Bridge: converts pipeline callbacks into Textual messages
# ---------------------------------------------------------------------------


class PipelineBridge:
    """Thread-safe bridge: pipeline callbacks → Textual messages.

    Instantiate on the main thread, then pass :meth:`on_clip_start` and
    :meth:`on_progress` as callback arguments to pipeline functions.

    *target* is the widget that should receive the messages (typically the
    panel with the ``@on()`` handlers).  Defaults to *app* for backwards
    compatibility, but callers should always pass the panel.
    """

    _throttle: float = 0.05  # 50 ms between intermediate progress posts

    def __init__(self, app: App, target: object | None = None) -> None:
        self._app = app
        self._target = target or app
        self._last_progress: float = 0.0  # monotonic timestamp

    def on_clip_start(self, clip_name: str, total: int) -> None:
        self._post(ClipStarted(clip_name, total))

    def on_progress(self, done: int, total: int, *args: int) -> None:
        now = time.monotonic()
        # Always post the final frame; throttle intermediate updates.
        if done < total and (now - self._last_progress) < self._throttle:
            return
        self._last_progress = now
        bytes_read = args[0] if len(args) > 0 else 0
        bytes_written = args[1] if len(args) > 1 else 0
        self._post(ProgressUpdate(done, total, bytes_read, bytes_written))

    def _post(self, message: object) -> None:
        """Non-blocking post from a worker thread to the Textual event loop."""
        try:
            loop = self._app._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._target.post_message, message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TUI log handler: routes log records to the Textual log pane
# ---------------------------------------------------------------------------


class TUILogHandler(logging.Handler):
    """Routes log records to the TUI's log pane via :class:`LogMessage`.

    The handler starts *inactive* and only forwards records while
    :attr:`active` is ``True``.  Each panel sets ``active = True`` when
    its worker starts and ``active = False`` when it finishes.  Because
    only one GPU operation runs at a time, this keeps logs in the correct
    panel without needing fragile thread-ID filtering.
    """

    def __init__(self, app: App, target: object | None = None) -> None:
        super().__init__(level=logging.DEBUG)
        self._app = app
        self._target = target or app
        self._posting = False  # re-entrancy guard against feedback loops
        self.active = False

    def emit(self, record: logging.LogRecord) -> None:
        if not self.active:
            return
        if self._posting:
            return  # prevent feedback loop caused by logging inside emit
        self._posting = True
        try:
            msg = self.format(record)
            loop = self._app._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(
                    self._target.post_message,
                    LogMessage(msg, record.levelno),
                )
        except Exception:
            self.handleError(record)
        finally:
            self._posting = False
