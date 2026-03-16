"""CorridorKey Engine — server process.

The server reads JSON-RPC messages from a transport, dispatches them,
and writes responses.  Jobs run in a dedicated thread while the main
loop continues handling requests (status queries, cancellation, etc.).

Usage::

    corridorkey-server                    # stdio (default)
    corridorkey-server --listen :9400     # TCP daemon
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
import uuid
from typing import Any

from ck_engine.transport import Transport, TransportClosed

logger = logging.getLogger(__name__)


class EngineServer:
    """Main engine server."""

    def __init__(self, transport: Transport) -> None:
        from ck_engine.engine.event_bus import EventBus
        from ck_engine.engine.model_pool import ModelPool
        from ck_engine.engine.job_runner import JobRunner
        from ck_engine.engine.dispatcher import Dispatcher

        self.transport = transport
        self.event_bus = EventBus(transport)
        self.model_pool = ModelPool(self.event_bus)
        self.job_runner = JobRunner(self.model_pool, self.event_bus)
        self.dispatcher = Dispatcher(self)

        self._shutdown = False
        self._start_time = time.monotonic()

        # Active job tracking
        self._active_job_id: str | None = None
        self._active_job_type: str = ""
        self._active_job_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._job_lock = threading.Lock()

    @property
    def state(self) -> str:
        if self._shutdown:
            return "shutting_down"
        if self._active_job_id is not None:
            return "busy"
        return "idle"

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time

    def start_job(self, job_type: str, params: Any) -> str:
        """Start a job in a background thread. Returns job_id."""
        with self._job_lock:
            job_id = f"j-{uuid.uuid4().hex[:8]}"
            cancel = threading.Event()
            self._cancel_event = cancel
            self._active_job_id = job_id
            self._active_job_type = job_type

            def _run() -> None:
                try:
                    # Install log handler so pipeline logs become events
                    self.event_bus.install_log_handler(level=logging.INFO)

                    if job_type == "generate":
                        self.job_runner.run_generate(job_id, params, cancel)
                    else:
                        self.job_runner.run_inference(job_id, params, cancel)
                except Exception as exc:
                    import traceback
                    err_msg = str(exc) or f"{type(exc).__name__}: {exc!r}"
                    tb = traceback.format_exc()
                    logger.error("Job %s failed: %s\n%s", job_id, err_msg, tb)
                    from ck_engine.api.events import JobFailed
                    self.event_bus.emit(JobFailed(job_id=job_id, error=err_msg))
                finally:
                    self.event_bus.remove_log_handler()
                    with self._job_lock:
                        self._active_job_id = None
                        self._active_job_thread = None
                        self._cancel_event = None

            thread = threading.Thread(target=_run, name=f"job-{job_id}", daemon=True)
            self._active_job_thread = thread
            thread.start()

            return job_id

    def cancel_job(self) -> None:
        """Signal the active job to cancel."""
        with self._job_lock:
            if self._cancel_event is not None:
                self._cancel_event.set()

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a job. Returns None if not found."""
        with self._job_lock:
            if self._active_job_id == job_id:
                return {
                    "job_id": job_id,
                    "state": "running",
                    "type": self._active_job_type,
                    "current_clip": "",
                    "progress": {"done": 0, "total": 0},
                    "clips_completed": 0,
                    "clips_total": 0,
                    "elapsed_seconds": 0.0,
                }
        return None

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown = True
        self.cancel_job()

    def run(self) -> None:
        """Main server loop. Blocks until shutdown."""
        logger.info("CorridorKey engine started (transport: %s)", type(self.transport).__name__)

        while not self._shutdown:
            # Drain pending events
            self.event_bus.drain()

            # Read next message
            try:
                message = self.transport.read_message()
            except TransportClosed:
                logger.info("Transport closed")
                break
            except Exception as exc:
                logger.warning("Transport read error: %s", exc)
                continue

            if message is None:
                logger.info("Client disconnected (EOF)")
                break  # Exit run loop — caller decides whether to accept another connection

            # Dispatch and respond
            response = self.dispatcher.dispatch(message)

            # Drain events that may have been generated during dispatch
            self.event_bus.drain()

            if response is not None:
                try:
                    self.transport.write_message(response)
                except TransportClosed:
                    logger.info("Transport closed during write")
                    break
                except Exception as exc:
                    logger.warning("Transport write error: %s", exc)

        # Shutdown
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up after a client session ends.

        Cancels any active job and closes the transport, but does NOT
        unload models — in daemon mode they persist across connections.
        """
        # Cancel active job
        self.cancel_job()

        # Wait for job thread to finish (with timeout)
        thread = self._active_job_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Job thread did not exit in 5s")

        # Remove log handler
        self.event_bus.remove_log_handler()

        # Close transport (but not model pool)
        try:
            self.transport.close()
        except Exception:
            pass


def main() -> None:
    """CLI entry point for the engine process."""
    parser = argparse.ArgumentParser(
        prog="corridorkey-server",
        description="CorridorKey Engine — inference server process",
    )
    parser.add_argument(
        "--listen",
        metavar="ADDRESS",
        help="TCP address for daemon mode (e.g., :9400 or 0.0.0.0:9400)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging to stderr (stdout is used for stdio transport)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Handle signals for graceful shutdown
    _daemon_shutdown = threading.Event()

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        _daemon_shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.listen:
        # TCP daemon mode — accept connections in a loop.  When a client
        # disconnects, the server goes back to listening for the next one.
        # Models stay resident in VRAM across connections.
        import socket as _socket

        from ck_engine.transport.tcp import TcpTransport, _parse_address

        host, port = _parse_address(args.listen)
        server_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        server_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)  # poll for shutdown signal
        server_sock.bind((host, port))
        server_sock.listen(1)
        logger.info("Listening on %s:%d ...", host, port)

        # Persistent model pool shared across connections
        from ck_engine.engine.event_bus import EventBus
        from ck_engine.engine.model_pool import ModelPool

        shared_model_pool: ModelPool | None = None

        try:
            while not _daemon_shutdown.is_set():
                try:
                    conn, addr = server_sock.accept()
                except TimeoutError:
                    continue
                logger.info("Client connected from %s:%d", *addr)
                transport = TcpTransport(conn)
                server = EngineServer(transport)
                # Reuse models from previous connections
                if shared_model_pool is not None:
                    server.model_pool = shared_model_pool
                    server.job_runner._model_pool = shared_model_pool
                server.run()
                # Preserve model pool for next connection
                shared_model_pool = server.model_pool
                logger.info("Client disconnected, waiting for next connection...")
        finally:
            server_sock.close()
            if shared_model_pool is not None:
                shared_model_pool.unload("all")
            logger.info("Daemon stopped")
    else:
        # Stdio mode — single connection, exit when client disconnects.
        from ck_engine.transport.stdio import StdioTransport

        transport = StdioTransport(sys.stdin.buffer, sys.stdout.buffer)
        server = EngineServer(transport)
        server.run()


if __name__ == "__main__":
    main()
