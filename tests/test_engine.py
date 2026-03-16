"""Tests for the CorridorKey engine: EventBus, ModelPool, Dispatcher, Server.

All tests run without a GPU or real model loading. A MockTransport backed
by queue.Queue objects stands in for stdio/TCP transports.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import pytest

from ck_engine.api.errors import INVALID_PATH, INVALID_REQUEST, METHOD_NOT_FOUND
from ck_engine.api.events import (
    EVENT_CATEGORIES,
    JobProgress,
    LogEvent,
    ModelLoaded,
)
from ck_engine.engine.event_bus import EventBus
from ck_engine.engine.model_pool import ModelPool
from ck_engine.engine.server import EngineServer
from ck_engine.transport import Transport, TransportClosed


# ---------------------------------------------------------------------------
# MockTransport
# ---------------------------------------------------------------------------


class MockTransport(Transport):
    """Transport backed by queues for testing."""

    def __init__(self) -> None:
        self._inbox: queue.Queue[dict | None] = queue.Queue()  # messages TO the server
        self._outbox: queue.Queue[dict] = queue.Queue()  # messages FROM the server
        self._closed = False

    # -- test helpers -------------------------------------------------------

    def send_to_server(self, msg: dict) -> None:
        """Test helper: enqueue a message for the server to read."""
        self._inbox.put(msg)

    def send_eof(self) -> None:
        """Test helper: signal EOF."""
        self._inbox.put(None)

    def receive_from_server(self, timeout: float = 2.0) -> dict:
        """Test helper: get the next message written by the server."""
        return self._outbox.get(timeout=timeout)

    # -- Transport interface ------------------------------------------------

    def read_message(self) -> dict | None:
        if self._closed:
            raise TransportClosed("closed")
        msg = self._inbox.get()
        if msg is None:
            return None
        return msg

    def write_message(self, msg: dict) -> None:
        if self._closed:
            raise TransportClosed("closed")
        self._outbox.put(msg)

    def close(self) -> None:
        self._closed = True

    @property
    def is_open(self) -> bool:
        return not self._closed


# ---------------------------------------------------------------------------
# TestEventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    """Tests for backend.engine.event_bus.EventBus."""

    def _make_bus(self) -> tuple[EventBus, MockTransport]:
        transport = MockTransport()
        bus = EventBus(transport)
        return bus, transport

    def test_emit_writes_to_transport(self) -> None:
        bus, transport = self._make_bus()
        event = JobProgress(job_id="j-1", clip="clip_a", done=5, total=10)
        bus.emit(event)
        bus.drain()
        msg = transport.receive_from_server(timeout=2.0)
        assert msg["method"] == "event.job.progress"
        assert msg["params"]["job_id"] == "j-1"
        assert msg["params"]["done"] == 5

    def test_subscription_filtering(self) -> None:
        bus, transport = self._make_bus()
        # Replace default "all" with only "job"
        bus.unsubscribe(["all"])
        bus.subscribe(["job"])

        # model event should be filtered out
        bus.emit(ModelLoaded(model="inf", device="cpu", vram_mb=0.0, load_seconds=0.1))
        bus.drain()
        assert transport._outbox.empty()

        # job event should pass through
        bus.emit(JobProgress(job_id="j-2", clip="c", done=1, total=2))
        bus.drain()
        msg = transport.receive_from_server(timeout=2.0)
        assert msg["method"] == "event.job.progress"

    def test_subscribe_all_default(self) -> None:
        bus, transport = self._make_bus()
        # Default subscription includes "all" — every category should arrive
        bus.emit(ModelLoaded(model="gen", device="cpu", vram_mb=0.0, load_seconds=0.0))
        bus.drain()
        msg = transport.receive_from_server(timeout=2.0)
        assert msg["method"] == "event.model.loaded"

    def test_unsubscribe(self) -> None:
        bus, transport = self._make_bus()
        # Start with "all", unsubscribe from "log"
        bus.unsubscribe(["log"])

        # log event should still arrive because "all" is still present
        bus.emit(LogEvent(level="info", message="test"))
        bus.drain()
        msg = transport.receive_from_server(timeout=2.0)
        assert msg["method"] == "event.log"

        # Now remove "all" and add only "job"
        bus.unsubscribe(["all"])
        bus.subscribe(["job"])

        # log event should NOT arrive
        bus.emit(LogEvent(level="info", message="dropped"))
        bus.drain()
        assert transport._outbox.empty()

        # job event SHOULD arrive
        bus.emit(JobProgress(job_id="j-3", clip="c", done=0, total=1))
        bus.drain()
        msg = transport.receive_from_server(timeout=2.0)
        assert msg["method"] == "event.job.progress"

    def test_log_handler(self) -> None:
        bus, transport = self._make_bus()
        bus.install_log_handler(level=logging.INFO)
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.DEBUG)
        try:
            test_logger = logging.getLogger("test.engine.log_handler")
            test_logger.info("hello")
            bus.drain()
            msg = transport.receive_from_server(timeout=2.0)
            assert msg["method"] == "event.log"
            assert "hello" in msg["params"]["message"]
        finally:
            bus.remove_log_handler()
            root.setLevel(old_level)

    def test_thread_safe_emit(self) -> None:
        bus, transport = self._make_bus()
        barrier = threading.Barrier(10)

        def _emit_one(idx: int) -> None:
            barrier.wait()
            bus.emit(JobProgress(job_id=f"j-{idx}", clip="c", done=idx, total=10))

        threads = [threading.Thread(target=_emit_one, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        bus.drain()
        received = []
        while not transport._outbox.empty():
            received.append(transport.receive_from_server(timeout=1.0))
        assert len(received) == 10


# ---------------------------------------------------------------------------
# TestModelPool
# ---------------------------------------------------------------------------


class TestModelPool:
    """Tests for backend.engine.model_pool.ModelPool (no GPU)."""

    def test_status_empty(self) -> None:
        pool = ModelPool()
        status = pool.status()
        assert status["inference_engine"] is None
        assert status["generator"] is None

    def test_unload_empty(self) -> None:
        pool = ModelPool()
        freed = pool.unload("all")
        assert freed == 0.0

    def test_hash_config_deterministic(self) -> None:
        h1 = ModelPool._hash_config("torch", "cuda:0", 2048, None)
        h2 = ModelPool._hash_config("torch", "cuda:0", 2048, None)
        assert h1 == h2

    def test_hash_config_differs(self) -> None:
        h1 = ModelPool._hash_config("torch", "cuda:0", 2048, None)
        h2 = ModelPool._hash_config("torch", "cuda:0", 1024, None)
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestDispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Tests for backend.engine.dispatcher.Dispatcher."""

    def _make_server(self) -> tuple[EngineServer, MockTransport]:
        transport = MockTransport()
        server = EngineServer(transport)
        return server, transport

    def _dispatch(self, server: EngineServer, msg: dict) -> dict | None:
        return server.dispatcher.dispatch(msg)

    def test_capabilities(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "engine.capabilities", "id": 1,
        })
        assert resp is not None
        assert "result" in resp
        result = resp["result"]
        assert "version" in result
        assert "generators" in result
        assert "backends" in result
        assert "devices" in result

    def test_unknown_method(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "no.such.method", "id": 2,
        })
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_invalid_request_no_jsonrpc(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {"method": "engine.status", "id": 3})
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_invalid_request_no_method(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {"jsonrpc": "2.0", "id": 4})
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_status_idle(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "engine.status", "id": 5,
        })
        assert resp is not None
        result = resp["result"]
        assert result["state"] == "idle"

    def test_shutdown(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "engine.shutdown", "id": 6,
        })
        assert resp is not None
        assert resp["result"] == "ok"
        assert server._shutdown is True

    def test_job_submit_no_path(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "job.submit", "id": 7,
            "params": {"type": "generate", "path": "/nonexistent/path"},
        })
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == INVALID_PATH

    def test_subscribe_unsubscribe(self) -> None:
        server, _ = self._make_server()

        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "events.subscribe", "id": 8,
            "params": {"categories": ["job"]},
        })
        assert resp is not None
        assert resp["result"] == "ok"

        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "events.unsubscribe", "id": 9,
            "params": {"categories": ["job"]},
        })
        assert resp is not None
        assert resp["result"] == "ok"

    def test_model_status(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "model.status", "id": 10,
        })
        assert resp is not None
        result = resp["result"]
        assert "inference_engine" in result
        assert "generator" in result

    def test_model_unload(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "model.unload", "id": 11,
            "params": {"which": "all"},
        })
        assert resp is not None
        result = resp["result"]
        assert "freed_mb" in result
        assert result["freed_mb"] == 0.0

    def test_project_scan_invalid_path(self) -> None:
        server, _ = self._make_server()
        resp = self._dispatch(server, {
            "jsonrpc": "2.0", "method": "project.scan", "id": 12,
            "params": {"path": "/definitely/not/a/real/path"},
        })
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == INVALID_PATH


# ---------------------------------------------------------------------------
# TestServer (integration)
# ---------------------------------------------------------------------------


def _run_server_test(messages: list[dict]) -> list[dict]:
    """Send messages to a server in a thread, collect responses."""
    transport = MockTransport()
    server = EngineServer(transport)

    # Queue all messages + EOF
    for msg in messages:
        transport.send_to_server(msg)
    transport.send_eof()

    # Run server in thread
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Collect responses
    responses: list[dict] = []
    while t.is_alive() or not transport._outbox.empty():
        try:
            resp = transport.receive_from_server(timeout=2.0)
            responses.append(resp)
        except queue.Empty:
            break

    t.join(timeout=3.0)
    return responses


class TestServer:
    """Integration tests that run the full EngineServer in a thread."""

    def test_server_capabilities(self) -> None:
        responses = _run_server_test([
            {"jsonrpc": "2.0", "method": "engine.capabilities", "id": 1},
        ])
        result_resp = next(r for r in responses if r.get("id") == 1)
        assert "result" in result_resp
        assert "version" in result_resp["result"]

    def test_server_shutdown(self) -> None:
        responses = _run_server_test([
            {"jsonrpc": "2.0", "method": "engine.shutdown", "id": 1},
        ])
        result_resp = next(r for r in responses if r.get("id") == 1)
        assert result_resp["result"] == "ok"

    def test_server_status(self) -> None:
        responses = _run_server_test([
            {"jsonrpc": "2.0", "method": "engine.status", "id": 1},
        ])
        result_resp = next(r for r in responses if r.get("id") == 1)
        assert "result" in result_resp
        assert result_resp["result"]["state"] == "idle"

    def test_server_multiple_requests(self) -> None:
        responses = _run_server_test([
            {"jsonrpc": "2.0", "method": "engine.capabilities", "id": 1},
            {"jsonrpc": "2.0", "method": "engine.status", "id": 2},
            {"jsonrpc": "2.0", "method": "engine.shutdown", "id": 3},
        ])
        # Find each response by id
        ids_found = {r["id"] for r in responses if "id" in r}
        assert {1, 2, 3}.issubset(ids_found)

    def test_server_project_scan_with_temp_dir(self, tmp_path) -> None:
        """Scan a temp project directory via the server."""
        # Create a minimal clip structure
        clip_dir = tmp_path / "clips" / "test_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        # Create a dummy frame
        (input_dir / "00001.png").write_bytes(b"fake png")

        responses = _run_server_test([
            {"jsonrpc": "2.0", "method": "project.scan", "id": 1,
             "params": {"path": str(tmp_path)}},
        ])

        # Find the response (skip any event notifications)
        result_resp = next(r for r in responses if r.get("id") == 1)
        assert "result" in result_resp
        result = result_resp["result"]
        assert result["project_path"] == str(tmp_path)
        assert len(result["clips"]) >= 1
        clip = result["clips"][0]
        assert clip["name"] == "test_clip"
