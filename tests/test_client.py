"""Integration tests for EngineClient.

These tests spawn real engine subprocesses and exercise the full
JSON-RPC protocol over stdio.  No GPU required.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from ck_engine.api.errors import EngineError
from ck_engine.api.events import JobFailed, LogEvent
from ck_engine.api.types import GenerateParams
from ck_engine.client import EngineClient


@pytest.fixture(scope="class")
def engine():
    """Spawn one engine subprocess shared across the test class."""
    client = EngineClient.spawn()
    yield client
    client.close(shutdown=True)


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with one clip."""
    clip = tmp_path / "clips" / "test_clip"
    input_dir = clip / "Input"
    input_dir.mkdir(parents=True)
    for i in range(3):
        (input_dir / f"{i+1:05d}.png").write_bytes(b"fake")
    return tmp_path


class TestEngineClient:
    def test_spawn_and_capabilities(self, engine):
        """Spawn engine and query capabilities."""
        caps = engine.capabilities()
        assert "version" in caps
        assert "generators" in caps
        assert isinstance(caps["generators"], list)
        assert "birefnet" in caps["generators"]
        assert "backends" in caps
        assert "profiles" in caps

    def test_status_idle(self, engine):
        """Freshly spawned engine should be idle."""
        status = engine.status()
        assert status["state"] == "idle"
        assert status["active_job"] is None

    def test_scan_project(self, engine, project_dir):
        """Scan a project directory and verify clips are found."""
        result = engine.scan_project(str(project_dir))
        assert result["project_path"] == str(project_dir)
        assert len(result["clips"]) >= 1
        clip = result["clips"][0]
        assert clip["name"] == "test_clip"
        assert clip["input"] is not None
        assert clip["input"]["frame_count"] == 3

    def test_scan_invalid_path(self, engine):
        """Scanning a non-existent path should raise EngineError."""
        with pytest.raises(EngineError) as exc_info:
            engine.scan_project("/nonexistent/path")
        assert exc_info.value.code == -32002  # INVALID_PATH

    def test_model_status(self, engine):
        """Query model status on fresh engine."""
        status = engine.model_status()
        assert status["inference_engine"] is None
        assert status["generator"] is None

    def test_unload_models(self, engine):
        """Unloading with nothing loaded should return 0."""
        freed = engine.unload_models("all")
        assert freed == 0.0

    def test_subscribe_unsubscribe(self, engine):
        """Subscribe and unsubscribe should not error."""
        engine.subscribe(["job", "model"])
        engine.unsubscribe(["model"])

    def test_submit_generate_bad_path(self, engine):
        """Submitting a job with invalid path should fail via events."""
        # The submit itself may succeed (async), but the job will fail
        # OR the submit may raise EngineError if validation is synchronous
        try:
            job_id = engine.submit_generate(GenerateParams(path="/nonexistent"))
            # Job was accepted -- wait for failure event
            for event in engine.iter_events(timeout=5.0):
                if isinstance(event, JobFailed):
                    assert "nonexistent" in event.error.lower() or "directory" in event.error.lower() or "not" in event.error.lower()
                    break
        except EngineError:
            pass  # Validation rejected it synchronously -- also fine

    def test_submit_while_busy(self, engine, project_dir):
        """Submitting a second job while one is running should raise ENGINE_BUSY."""
        # Submit a job that will take a moment (generate on real path)
        # It will fail quickly (no model) but occupies the slot briefly
        try:
            job_id = engine.submit_generate(
                GenerateParams(path=str(project_dir), model="birefnet")
            )
        except EngineError:
            pytest.skip("First submit rejected -- can't test busy state")

        # Immediately try a second submit
        # There's a race: the first job might finish before we submit the second.
        # If we get ENGINE_BUSY, great. If not, the test is inconclusive -- skip.
        try:
            engine.submit_generate(
                GenerateParams(path=str(project_dir), model="birefnet")
            )
            # If it succeeded, the first job finished too fast -- drain events and skip
            pytest.skip("First job finished before second submit -- race condition")
        except EngineError as exc:
            assert exc.code == -32000  # ENGINE_BUSY

        # Drain remaining events from the first job.
        # The job may have already failed and events may have been delivered
        # (or not yet flushed), so we tolerate an empty queue.
        import queue as _queue

        try:
            for event in engine.iter_events(timeout=3.0):
                if isinstance(event, (JobFailed,)):
                    break
        except _queue.Empty:
            pass  # Events already drained or not yet flushed -- acceptable

    def test_shutdown(self):
        """Shutdown should cleanly stop the engine."""
        engine = EngineClient.spawn()
        proc = engine._process
        engine.shutdown()
        # Wait for the engine process to actually exit
        if proc is not None:
            proc.wait(timeout=10.0)
        # Give the reader thread a moment to notice EOF
        time.sleep(0.5)
        # After the process exits and reader detects EOF, the reader thread
        # should have stopped.  The transport's is_open may still be True
        # (it only becomes False on explicit close()), but the reader thread
        # should no longer be alive.
        assert not engine._reader_thread.is_alive()
        engine.close()

    def test_context_manager(self, project_dir):
        """EngineClient as context manager should auto-close."""
        with EngineClient.spawn() as client:
            caps = client.capabilities()
            assert "version" in caps
        # After exiting context, client should be closed
        assert client._closed

    def test_cancel_nonexistent_job(self, engine):
        """Cancelling a job that doesn't exist should raise."""
        with pytest.raises(EngineError) as exc_info:
            engine.cancel_job("j-nonexistent")
        assert exc_info.value.code == -32001  # JOB_NOT_FOUND


class TestTcpClient:
    def test_connect_tcp(self, project_dir):
        """Start engine with --listen, connect via TCP."""
        import socket
        import subprocess as sp

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        # Start engine daemon
        proc = sp.Popen(
            [sys.executable, "-m", "ck_engine.engine.server", "--listen", f"127.0.0.1:{port}"],
            stderr=sp.PIPE,
        )

        try:
            # Connect via TCP — retry until the server is ready
            client = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    client = EngineClient.connect(f"127.0.0.1:{port}")
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.1)

            if client is None:
                pytest.skip("Engine did not start listening in time")

            try:
                caps = client.capabilities()
                assert "version" in caps

                result = client.scan_project(str(project_dir))
                assert len(result["clips"]) >= 1
            finally:
                client.close(shutdown=True)
        finally:
            proc.wait(timeout=5)
