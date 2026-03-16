"""CorridorKey engine client -- connect to an engine process.

The client provides a Pythonic interface to the JSON-RPC engine protocol.
Frontends use this to spawn or connect to an engine, submit jobs, and
receive structured events.

Usage::

    # Spawn a local engine (stdio)
    client = EngineClient.spawn()
    caps = client.capabilities()

    # Submit a job and iterate events
    job_id = client.submit_generate(GenerateParams(path="/project"))
    for event in client.iter_events():
        if isinstance(event, JobCompleted):
            break

    # Connect to a remote daemon (TCP)
    client = EngineClient.connect("192.168.1.10:9400")
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, Iterator

from ck_engine.api.errors import EngineError
from ck_engine.api.events import ALL_EVENT_TYPES, Event
from ck_engine.api.types import (
    ClipInfo,
    EngineCapabilities,
    EngineStatus,
    GenerateParams,
    InferenceParams,
    JobResult,
    JobStatus,
)
from ck_engine.transport import Transport, TransportClosed

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ResponseFuture:
    """Waits for a JSON-RPC response with a specific id."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: dict | None = None

    def set(self, response: dict) -> None:
        self._result = response
        self._event.set()

    def wait(self, timeout: float = 30.0) -> dict:
        if not self._event.wait(timeout):
            raise TimeoutError("No response from engine within timeout")
        assert self._result is not None
        return self._result


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EngineClient:
    """Client for the CorridorKey engine process."""

    def __init__(
        self, transport: Transport, process: subprocess.Popen | None = None
    ) -> None:
        self._transport = transport
        self._process = process  # Only set when we spawned the engine
        self._next_id = 0
        self._id_lock = threading.Lock()

        # Response dispatch
        self._pending: dict[int | str, _ResponseFuture] = {}
        self._pending_lock = threading.Lock()

        # Event queue for iter_events()
        self._events: queue.Queue[Event | None] = queue.Queue()

        # Reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="engine-client-reader", daemon=True
        )
        self._closed = False
        self._reader_thread.start()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def spawn(cls, **kwargs: Any) -> EngineClient:
        """Start a new engine subprocess and connect via stdio.

        Keyword arguments are passed as CLI flags to the engine process.
        For example: ``spawn(log_level="DEBUG")``.
        """
        # Locate the engine executable: check the same venv as the current
        # Python, then PATH, then fall back to running the module directly.
        import os
        import shutil

        venv_bin = os.path.join(os.path.dirname(sys.executable), "corridorkey-engine")
        if os.path.isfile(venv_bin):
            cmd = [venv_bin, "serve"]
        elif shutil.which("corridorkey-engine"):
            cmd = ["corridorkey-engine", "serve"]
        else:
            cmd = [sys.executable, "-m", "ck_engine.engine.server"]

        log_level = kwargs.pop("log_level", None)
        if log_level:
            cmd.extend(["--log-level", log_level])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Discard diagnostic logs (events carry structured info)
        )

        from ck_engine.transport.stdio import StdioTransport

        transport = StdioTransport(proc.stdout, proc.stdin)  # type: ignore[arg-type]

        return cls(transport, process=proc)

    @classmethod
    def connect(cls, address: str) -> EngineClient:
        """Connect to a running engine daemon via TCP."""
        from ck_engine.transport.tcp import TcpTransport

        transport = TcpTransport.connect(address)
        return cls(transport)

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: reads messages, dispatches responses and events."""
        while not self._closed:
            try:
                msg = self._transport.read_message()
            except TransportClosed:
                break
            except Exception as exc:
                logger.debug("Reader error: %s", exc)
                break

            if msg is None:
                break  # EOF

            # Is it a response? (has "id" field)
            msg_id = msg.get("id")
            if msg_id is not None:
                with self._pending_lock:
                    future = self._pending.pop(msg_id, None)
                if future is not None:
                    future.set(msg)
                continue

            # Is it a notification? (has "method" but no "id")
            method = msg.get("method", "")
            if method:
                event = self._parse_event(msg)
                if event is not None:
                    self._events.put(event)

        # Signal end of events
        self._events.put(None)

    @staticmethod
    def _parse_event(msg: dict) -> Event | None:
        """Parse a JSON-RPC notification into an Event object."""
        method = msg.get("method", "")
        params = msg.get("params", {})

        for event_cls in ALL_EVENT_TYPES:
            if event_cls.method == method:
                try:
                    return event_cls(**params)
                except Exception:
                    logger.debug("Failed to parse event %s: %s", method, params)
                    return None

        logger.debug("Unknown event method: %s", method)
        return None

    # ------------------------------------------------------------------
    # Request / response helpers
    # ------------------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _request(
        self, method: str, params: dict | None = None, timeout: float = 30.0
    ) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        req_id = self._next_request_id()

        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
        }
        if params is not None:
            msg["params"] = params

        future = _ResponseFuture()
        with self._pending_lock:
            self._pending[req_id] = future

        try:
            self._transport.write_message(msg)
        except Exception:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise

        response = future.wait(timeout)

        # Check for error
        if "error" in response:
            err = response["error"]
            raise EngineError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )

        return response.get("result")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def capabilities(self) -> dict:
        """Query engine capabilities."""
        return self._request("engine.capabilities")

    def status(self) -> dict:
        """Query engine status."""
        return self._request("engine.status")

    def shutdown(self) -> None:
        """Request graceful engine shutdown."""
        try:
            self._request("engine.shutdown", timeout=5.0)
        except (TransportClosed, TimeoutError, OSError):
            pass  # Engine may close before we read the response

    def scan_project(self, path: str) -> dict:
        """Scan a project directory for clips.

        Returns dict with keys: project_path, is_v2, clips.
        """
        return self._request("project.scan", {"path": path})

    def submit_generate(self, params: GenerateParams) -> str:
        """Submit an alpha generation job. Returns job_id."""
        result = self._request("job.submit", {
            "type": "generate",
            **params.to_dict(),
        })
        return result["job_id"]

    def submit_inference(self, params: InferenceParams) -> str:
        """Submit an inference job. Returns job_id."""
        result = self._request("job.submit", {
            "type": "inference",
            **params.to_dict(),
        })
        return result["job_id"]

    def cancel_job(self, job_id: str) -> None:
        """Cancel the active job."""
        self._request("job.cancel", {"job_id": job_id})

    def job_status(self, job_id: str) -> dict:
        """Query a job's current status."""
        return self._request("job.status", {"job_id": job_id})

    def model_status(self) -> dict:
        """Query loaded model status."""
        return self._request("model.status")

    def unload_models(self, which: str = "all") -> float:
        """Unload models. Returns freed VRAM in MB."""
        result = self._request("model.unload", {"which": which})
        return result.get("freed_mb", 0.0)

    def subscribe(self, categories: list[str]) -> None:
        """Subscribe to event categories."""
        self._request("events.subscribe", {"categories": categories})

    def unsubscribe(self, categories: list[str]) -> None:
        """Unsubscribe from event categories."""
        self._request("events.unsubscribe", {"categories": categories})

    def iter_events(self, timeout: float | None = None) -> Iterator[Event]:
        """Iterate over incoming events.

        Blocks until an event arrives or the connection closes.
        Yields Event objects. Stops when the connection is closed.

        If timeout is set, raises ``queue.Empty`` after *timeout* seconds
        of no events.
        """
        while True:
            try:
                event = self._events.get(timeout=timeout)
            except queue.Empty:
                raise
            if event is None:
                return  # Connection closed
            yield event

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self, shutdown: bool = False) -> None:
        """Close the connection.

        If *shutdown* is True and we spawned the engine, send the shutdown
        command and wait for the process to exit.
        """
        if self._closed:
            return
        self._closed = True

        if shutdown:
            try:
                self.shutdown()
            except Exception:
                pass

        try:
            self._transport.close()
        except Exception:
            pass

        # Wait for reader thread
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)

        # Clean up subprocess
        if self._process is not None:
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
            self._process = None

    def __enter__(self) -> EngineClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(shutdown=self._process is not None)

    @property
    def is_connected(self) -> bool:
        """Whether the client is still connected to the engine."""
        return not self._closed and self._transport.is_open
