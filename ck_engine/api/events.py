"""Event types for the CorridorKey engine protocol.

Events are JSON-RPC notifications emitted by the engine to subscribed
clients.  Each event class has a ``method`` class attribute (the
JSON-RPC method name) and a ``to_notification()`` method that produces
a complete JSON-RPC notification dict.
"""

from __future__ import annotations

import dataclasses
import time
from typing import ClassVar


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Event:
    """Base class for all engine events.

    Subclasses must be frozen dataclasses that also inherit from this class
    and define ``method`` and ``category`` as class-level ClassVar strings.
    """

    method: ClassVar[str]
    category: ClassVar[str]

    def _to_params(self) -> dict:
        """Return a plain dict of all fields on this event."""
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    def to_notification(self) -> dict:
        """Return a complete JSON-RPC 2.0 notification dict."""
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self._to_params(),
        }


# ---------------------------------------------------------------------------
# Job events
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JobAccepted(Event):
    """Emitted when the engine accepts a new job."""

    job_id: str
    type: str
    total_frames: int

    method: ClassVar[str] = "event.job.accepted"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class ClipStarted(Event):
    """Emitted when the engine begins processing a clip."""

    job_id: str
    clip: str
    frames: int
    clip_index: int
    clips_total: int

    method: ClassVar[str] = "event.job.clip_started"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class JobProgress(Event):
    """Emitted periodically while processing frames within a clip."""

    job_id: str
    clip: str
    done: int
    total: int
    bytes_read: int = 0
    bytes_written: int = 0
    fps: float = 0.0

    method: ClassVar[str] = "event.job.progress"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class ClipCompleted(Event):
    """Emitted when all frames in a clip have been processed."""

    job_id: str
    clip: str
    frames_ok: int
    frames_failed: int

    method: ClassVar[str] = "event.job.clip_completed"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class JobCompleted(Event):
    """Emitted when the engine finishes all clips in a job.

    ``failed_frames`` is a list of ``{"clip": str, "frame": int, "error": str}``
    dicts, or ``None`` if there were no failures.
    """

    job_id: str
    clips_ok: int
    clips_failed: int
    total_frames: int
    frames_ok: int
    frames_failed: int
    elapsed_seconds: float
    failed_frames: list[dict] | None = None

    method: ClassVar[str] = "event.job.completed"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class JobFailed(Event):
    """Emitted when a job terminates with an unrecoverable error."""

    job_id: str
    error: str

    method: ClassVar[str] = "event.job.failed"
    category: ClassVar[str] = "job"


@dataclasses.dataclass(frozen=True)
class JobCancelled(Event):
    """Emitted when a job is cancelled before completion."""

    job_id: str
    frames_completed: int

    method: ClassVar[str] = "event.job.cancelled"
    category: ClassVar[str] = "job"


# ---------------------------------------------------------------------------
# Model events
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelLoading(Event):
    """Emitted when the engine begins loading a model checkpoint."""

    model: str
    device: str

    method: ClassVar[str] = "event.model.loading"
    category: ClassVar[str] = "model"


@dataclasses.dataclass(frozen=True)
class ModelLoaded(Event):
    """Emitted when a model has been successfully loaded and is ready."""

    model: str
    device: str
    vram_mb: float
    load_seconds: float

    method: ClassVar[str] = "event.model.loaded"
    category: ClassVar[str] = "model"


@dataclasses.dataclass(frozen=True)
class ModelUnloaded(Event):
    """Emitted when a model is evicted from memory."""

    model: str
    freed_mb: float

    method: ClassVar[str] = "event.model.unloaded"
    category: ClassVar[str] = "model"


@dataclasses.dataclass(frozen=True)
class ModelRecompiling(Event):
    """Emitted when a model triggers a recompilation (e.g. shape change)."""

    reason: str
    backend: str

    method: ClassVar[str] = "event.model.recompiling"
    category: ClassVar[str] = "model"


# ---------------------------------------------------------------------------
# Log events
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LogEvent(Event):
    """Wraps a log record as a JSON-RPC notification."""

    level: str
    message: str
    logger: str = ""
    timestamp: float = dataclasses.field(default_factory=time.time)

    method: ClassVar[str] = "event.log"
    category: ClassVar[str] = "log"


# ---------------------------------------------------------------------------
# Warning events
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarningEvent(Event):
    """Non-fatal warning that clients may surface to users."""

    message: str
    clip: str = ""
    detail: str = ""

    method: ClassVar[str] = "event.warning"
    category: ClassVar[str] = "warning"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVENT_CATEGORIES: frozenset[str] = frozenset({"job", "model", "log", "warning", "all"})

ALL_EVENT_TYPES: tuple[type[Event], ...] = (
    JobAccepted,
    ClipStarted,
    JobProgress,
    ClipCompleted,
    JobCompleted,
    JobFailed,
    JobCancelled,
    ModelLoading,
    ModelLoaded,
    ModelUnloaded,
    ModelRecompiling,
    LogEvent,
    WarningEvent,
)
