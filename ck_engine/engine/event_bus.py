"""Event subscription and dispatch for the CorridorKey engine.

The EventBus sits between the job runner (which emits events on worker
threads) and the transport (which writes to the client).  It handles
subscription filtering and thread-safe queuing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from ck_engine.api.events import EVENT_CATEGORIES, Event, LogEvent

if TYPE_CHECKING:
    from ck_engine.transport import Transport


class EventBus:
    """Thread-safe event dispatcher with subscription filtering."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._subscribed: set[str] = {"all"}  # default: all events
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict] = queue.Queue()
        self._log_handler: _EventLogHandler | None = None

    def subscribe(self, categories: list[str]) -> None:
        """Add event categories to the subscription set."""
        with self._lock:
            for cat in categories:
                if cat in EVENT_CATEGORIES:
                    self._subscribed.add(cat)

    def unsubscribe(self, categories: list[str]) -> None:
        """Remove event categories from the subscription set."""
        with self._lock:
            for cat in categories:
                self._subscribed.discard(cat)

    def emit(self, event: Event) -> None:
        """Write an event to the transport immediately.  Thread-safe.

        The transport's write_message() is guarded by a write lock, so
        this is safe to call from any thread (job runner, log handler,
        etc.) without queuing or draining.
        """
        with self._lock:
            if "all" not in self._subscribed and event.category not in self._subscribed:
                return
        try:
            self._transport.write_message(event.to_notification())
        except Exception:
            pass  # transport closed — drop the event

    def drain(self) -> None:
        """No-op — events are now written immediately by emit().

        Kept for backwards compatibility with callers.
        """

    def flush_sync(self, event: Event) -> None:
        """Emit and immediately write an event.  Use for critical events
        (job.completed, job.failed) that must be sent before a response.

        Thread-safe — acquires the transport write lock internally.
        """
        with self._lock:
            if "all" not in self._subscribed and event.category not in self._subscribed:
                return
        try:
            self._transport.write_message(event.to_notification())
        except Exception:
            pass

    def install_log_handler(self, level: int = logging.INFO) -> None:
        """Attach a logging handler that converts log records to LogEvents."""
        if self._log_handler is not None:
            return
        self._log_handler = _EventLogHandler(self, level=level)
        logging.getLogger().addHandler(self._log_handler)

    def remove_log_handler(self) -> None:
        """Detach the logging handler."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    @property
    def pending_count(self) -> int:
        """Number of events in the queue."""
        return self._queue.qsize()


class _EventLogHandler(logging.Handler):
    """Converts Python log records to LogEvent and emits them."""

    def __init__(self, bus: EventBus, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._bus = bus
        self._posting = False  # re-entrancy guard

    def emit(self, record: logging.LogRecord) -> None:
        if self._posting:
            return
        self._posting = True
        try:
            event = LogEvent(
                level=record.levelname.lower(),
                message=self.format(record),
                logger=record.name,
                timestamp=record.created,
            )
            self._bus.emit(event)
        except Exception:
            self.handleError(record)
        finally:
            self._posting = False
