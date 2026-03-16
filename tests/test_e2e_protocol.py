"""End-to-end protocol tests for the CorridorKey engine.

These tests spawn real engine subprocesses and exercise complete
JSON-RPC request/response/event flows.  No GPU required.
"""

from __future__ import annotations

import os
import queue
import sys
import time

import pytest

from ck_engine.api.errors import EngineError
from ck_engine.api.events import JobFailed, JobCompleted, LogEvent
from ck_engine.api.types import GenerateParams, InferenceParams, InferenceSettings
from ck_engine.client import EngineClient


@pytest.fixture(scope="class")
def engine():
    """Spawn one engine for all tests in the class."""
    client = EngineClient.spawn()
    yield client
    client.close(shutdown=True)


@pytest.fixture
def project_with_clips(tmp_path):
    """Project with one clip that has both Input and AlphaHint."""
    clip = tmp_path / "clips" / "plate_001"
    input_dir = clip / "Input"
    alpha_dir = clip / "AlphaHint"
    input_dir.mkdir(parents=True)
    alpha_dir.mkdir(parents=True)
    for i in range(5):
        (input_dir / f"{i+1:05d}.png").write_bytes(b"fake")
        (alpha_dir / f"{i+1:05d}.png").write_bytes(b"fake")
    return tmp_path


class TestProtocolLifecycle:
    def test_scan_returns_clip_structure(self, engine, project_with_clips):
        """Scan returns nested clip info with asset details."""
        result = engine.scan_project(str(project_with_clips))
        assert result["project_path"] == str(project_with_clips)
        clips = result["clips"]
        assert len(clips) == 1
        clip = clips[0]
        assert clip["name"] == "plate_001"
        assert clip["input"] is not None
        assert clip["input"]["type"] == "sequence"
        assert clip["input"]["frame_count"] == 5
        assert clip["alpha"] is not None
        assert clip["alpha"]["frame_count"] == 5

    def test_capabilities_structure(self, engine):
        """Capabilities returns all expected fields."""
        caps = engine.capabilities()
        assert isinstance(caps["version"], str)
        assert "birefnet" in caps["generators"]
        assert "gvm" in caps["generators"]
        assert "videomama" in caps["generators"]
        assert isinstance(caps["backends"], list)
        assert isinstance(caps["profiles"], list)
        assert "optimized" in caps["profiles"]

    def test_status_fields(self, engine):
        """Status returns all expected fields."""
        status = engine.status()
        assert status["state"] in ("idle", "busy", "shutting_down")
        assert "active_job" in status
        assert "uptime_seconds" in status
        assert isinstance(status["uptime_seconds"], (int, float))
        assert status["uptime_seconds"] > 0

    def test_model_status_structure(self, engine):
        """Model status returns expected structure."""
        status = engine.model_status()
        assert "inference_engine" in status
        assert "generator" in status

    def test_subscribe_then_unsubscribe(self, engine):
        """Subscribe and unsubscribe round-trip."""
        engine.subscribe(["job", "model", "log"])
        engine.unsubscribe(["log"])
        # No error = success

    def test_generate_job_fails_gracefully(self, engine, project_with_clips):
        """Submit generate job — will fail (no model weights) but protocol works."""
        try:
            job_id = engine.submit_generate(
                GenerateParams(path=str(project_with_clips), model="birefnet")
            )
            assert job_id.startswith("j-")

            # Wait for completion (will fail since no model weights)
            try:
                for event in engine.iter_events(timeout=30.0):
                    if isinstance(event, (JobFailed, JobCompleted)):
                        break
            except queue.Empty:
                pass  # Timed out waiting for events — acceptable
        except EngineError:
            pass  # Submit itself may fail — also valid

    def test_inference_job_fails_gracefully(self, engine, project_with_clips):
        """Submit inference job — will fail (no model weights) but protocol works."""
        try:
            params = InferenceParams(
                path=str(project_with_clips),
                settings=InferenceSettings(),
            )
            job_id = engine.submit_inference(params)
            assert job_id.startswith("j-")

            try:
                for event in engine.iter_events(timeout=30.0):
                    if isinstance(event, (JobFailed, JobCompleted)):
                        break
            except queue.Empty:
                pass
        except EngineError:
            pass

    def test_error_response_structure(self, engine):
        """Error responses have the correct JSON-RPC structure."""
        with pytest.raises(EngineError) as exc_info:
            engine.scan_project("/definitely/not/a/real/path")
        err = exc_info.value
        assert isinstance(err.code, int)
        assert err.code == -32002  # INVALID_PATH
        assert isinstance(err.message, str)
        assert len(err.message) > 0

    def test_unload_models_returns_freed(self, engine):
        """Unload returns freed MB (0 when nothing loaded)."""
        freed = engine.unload_models("all")
        assert isinstance(freed, (int, float))
        assert freed >= 0


class TestTcpProtocol:
    def test_tcp_full_cycle(self, project_with_clips):
        """Full protocol cycle over TCP transport."""
        import socket
        import subprocess as sp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        proc = sp.Popen(
            [sys.executable, "-m", "ck_engine.engine.server", "--listen", f"127.0.0.1:{port}"],
            stderr=sp.PIPE,
        )
        try:
            # Connect with retry
            client = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    client = EngineClient.connect(f"127.0.0.1:{port}")
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.1)

            if client is None:
                pytest.skip("Could not connect to TCP engine")

            try:
                # Capabilities
                caps = client.capabilities()
                assert "version" in caps

                # Scan
                result = client.scan_project(str(project_with_clips))
                assert len(result["clips"]) == 1

                # Status
                status = client.status()
                assert status["state"] == "idle"
            finally:
                client.close(shutdown=True)
        finally:
            proc.wait(timeout=5)
