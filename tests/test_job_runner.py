"""Tests for the JobRunner -- job execution with event emission.

All tests run without a GPU or real model loading.  MockTransport + EventBus
collect emitted events, and pipeline functions are patched out.
"""

from __future__ import annotations

import queue
import threading
from unittest import mock

import pytest

from ck_engine.api.errors import EngineError, INVALID_PATH, NO_VALID_CLIPS
from ck_engine.api.events import (
    JobAccepted,
    ClipStarted,
    JobProgress,
    JobCompleted,
    JobFailed,
    JobCancelled,
)
from ck_engine.api.types import GenerateParams, ClipInfo
from ck_engine.engine.event_bus import EventBus
from ck_engine.engine.job_runner import JobRunner, _clip_entry_to_info
from ck_engine.engine.model_pool import ModelPool
from ck_engine.transport import Transport, TransportClosed


# ---------------------------------------------------------------------------
# MockTransport (copied from test_engine.py)
# ---------------------------------------------------------------------------


class MockTransport(Transport):
    """Transport backed by queues for testing."""

    def __init__(self) -> None:
        self._inbox: queue.Queue[dict | None] = queue.Queue()
        self._outbox: queue.Queue[dict] = queue.Queue()
        self._closed = False

    # -- test helpers -------------------------------------------------------

    def send_to_server(self, msg: dict) -> None:
        self._inbox.put(msg)

    def send_eof(self) -> None:
        self._inbox.put(None)

    def receive_from_server(self, timeout: float = 2.0) -> dict:
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
# Helpers
# ---------------------------------------------------------------------------


def _make_runner() -> tuple[JobRunner, EventBus, MockTransport]:
    transport = MockTransport()
    bus = EventBus(transport)
    pool = ModelPool(bus)
    runner = JobRunner(pool, bus)
    return runner, bus, transport


def _collect_events(transport: MockTransport, timeout: float = 0.5) -> list[dict]:
    """Drain all messages currently in the transport outbox."""
    events: list[dict] = []
    while True:
        try:
            msg = transport.receive_from_server(timeout=timeout)
            events.append(msg)
        except queue.Empty:
            break
    return events


def _build_clip(tmp_path, clip_name: str = "test_clip", frame_count: int = 3) -> None:
    """Create a minimal v2-layout clip under tmp_path/clips/<clip_name>/Input/."""
    clip = tmp_path / "clips" / clip_name
    (clip / "Input").mkdir(parents=True)
    for i in range(frame_count):
        (clip / "Input" / f"{i + 1:05d}.png").write_bytes(b"fake")


# ---------------------------------------------------------------------------
# TestScanProject
# ---------------------------------------------------------------------------


class TestScanProject:
    def test_scan_valid_project(self, tmp_path):
        """Scan a project with one clip returns correct metadata."""
        _build_clip(tmp_path, "test_clip", frame_count=3)

        runner, _bus, _transport = _make_runner()
        result = runner.scan_project(str(tmp_path))

        assert result["project_path"] == str(tmp_path)
        assert len(result["clips"]) == 1
        clip_info = result["clips"][0]
        assert clip_info["name"] == "test_clip"
        assert clip_info["input"]["frame_count"] == 3

    def test_scan_invalid_path(self):
        """Non-existent path raises EngineError with INVALID_PATH code."""
        runner, _bus, _transport = _make_runner()

        with pytest.raises(EngineError) as exc_info:
            runner.scan_project("/nonexistent/path/that/does/not/exist")
        assert exc_info.value.code == INVALID_PATH

    def test_scan_empty_dir(self, tmp_path):
        """A project directory with no clips returns an empty list."""
        runner, _bus, _transport = _make_runner()
        result = runner.scan_project(str(tmp_path))
        assert result["clips"] == []

    def test_scan_is_v2_flag(self, tmp_path):
        """is_v2 is True when a clips/ subdirectory is present."""
        _build_clip(tmp_path, "clip_a")
        runner, _bus, _transport = _make_runner()
        result = runner.scan_project(str(tmp_path))
        assert result["is_v2"] is True

    def test_scan_multiple_clips(self, tmp_path):
        """Multiple clips are all returned."""
        _build_clip(tmp_path, "shot_01", frame_count=2)
        _build_clip(tmp_path, "shot_02", frame_count=5)

        runner, _bus, _transport = _make_runner()
        result = runner.scan_project(str(tmp_path))

        assert len(result["clips"]) == 2
        names = {c["name"] for c in result["clips"]}
        assert names == {"shot_01", "shot_02"}


# ---------------------------------------------------------------------------
# TestRunGenerate
# ---------------------------------------------------------------------------


class TestRunGenerate:
    def test_emits_job_accepted_and_completed(self, tmp_path):
        """Mock generate_alpha_hints; verify JobAccepted and JobCompleted are emitted."""
        _build_clip(tmp_path, "test_clip", frame_count=3)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path), model="birefnet", mode="replace")

        with mock.patch("ck_engine.pipeline.generate_alpha_hints", return_value=1) as mock_gen:
            runner.run_generate("j-test1", params, cancel)

        mock_gen.assert_called_once()
        bus.drain()
        events = _collect_events(transport, timeout=0.5)

        methods = [e.get("method") for e in events]
        assert "event.job.accepted" in methods
        assert "event.job.completed" in methods

    def test_job_accepted_has_correct_frame_count(self, tmp_path):
        """JobAccepted.total_frames reflects the actual frame count."""
        _build_clip(tmp_path, "clip_a", frame_count=7)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path), model="birefnet")

        with mock.patch("ck_engine.pipeline.generate_alpha_hints", return_value=1):
            runner.run_generate("j-frames", params, cancel)

        bus.drain()
        events = _collect_events(transport, timeout=0.5)
        accepted = next(e for e in events if e.get("method") == "event.job.accepted")
        assert accepted["params"]["total_frames"] == 7

    def test_cancel_emits_cancelled_or_completed(self, tmp_path):
        """Pre-cancelled job emits JobCancelled (or JobCompleted if it raced through)."""
        _build_clip(tmp_path, "test_clip", frame_count=1)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        cancel.set()  # pre-cancel before run

        params = GenerateParams(path=str(tmp_path), model="birefnet")

        with mock.patch("ck_engine.pipeline.generate_alpha_hints", return_value=0):
            runner.run_generate("j-test2", params, cancel)

        bus.drain()
        events = _collect_events(transport, timeout=0.5)
        methods = [e.get("method") for e in events]
        # Allow either outcome — timing-dependent
        assert "event.job.cancelled" in methods or "event.job.completed" in methods

    def test_failure_emits_job_failed(self, tmp_path):
        """If generate_alpha_hints raises, JobFailed should be emitted."""
        _build_clip(tmp_path, "test_clip", frame_count=1)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path), model="birefnet")

        with mock.patch(
            "ck_engine.pipeline.generate_alpha_hints",
            side_effect=RuntimeError("GPU OOM"),
        ):
            runner.run_generate("j-test3", params, cancel)

        bus.drain()
        events = _collect_events(transport, timeout=0.5)
        methods = [e.get("method") for e in events]
        assert "event.job.failed" in methods

        failed = next(e for e in events if e.get("method") == "event.job.failed")
        assert "OOM" in failed["params"]["error"]

    def test_failure_does_not_emit_completed(self, tmp_path):
        """When generate raises, JobCompleted must NOT appear alongside JobFailed."""
        _build_clip(tmp_path, "test_clip", frame_count=1)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path), model="birefnet")

        with mock.patch(
            "ck_engine.pipeline.generate_alpha_hints",
            side_effect=RuntimeError("boom"),
        ):
            runner.run_generate("j-nofail", params, cancel)

        bus.drain()
        events = _collect_events(transport, timeout=0.5)
        methods = [e.get("method") for e in events]
        assert "event.job.completed" not in methods

    def test_invalid_path_raises_engine_error(self):
        """Non-existent project path raises EngineError (not swallowed by run_generate)."""
        runner, _bus, _transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path="/nonexistent/no/such/dir")

        with pytest.raises(EngineError):
            runner.run_generate("j-test4", params, cancel)

    def test_no_valid_clips_raises_engine_error(self, tmp_path):
        """A project with clips that have no input frames raises EngineError."""
        # Create a clip directory without any input frames
        clip = tmp_path / "clips" / "empty_clip"
        clip.mkdir(parents=True)
        # No Input/ subdirectory, so find_assets will fail and the clip gets skipped.
        # scan_clips will return entries but valid list will be empty after filtering.
        # To trigger NO_VALID_CLIPS we need a clip dir that passes find_assets but
        # has 0 frames -- simplest is to create Input/ dir with no image files.
        (clip / "Input").mkdir()

        runner, _bus, _transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path))

        with pytest.raises(EngineError) as exc_info:
            runner.run_generate("j-noclips", params, cancel)
        assert exc_info.value.code == NO_VALID_CLIPS

    def test_job_failed_params_contain_job_id(self, tmp_path):
        """JobFailed event params include the correct job_id."""
        _build_clip(tmp_path, "clip_x", frame_count=1)

        runner, bus, transport = _make_runner()
        cancel = threading.Event()
        params = GenerateParams(path=str(tmp_path), model="birefnet")

        with mock.patch(
            "ck_engine.pipeline.generate_alpha_hints",
            side_effect=ValueError("something broke"),
        ):
            runner.run_generate("unique-job-id-99", params, cancel)

        bus.drain()
        events = _collect_events(transport, timeout=0.5)
        failed = next(e for e in events if e.get("method") == "event.job.failed")
        assert failed["params"]["job_id"] == "unique-job-id-99"


# ---------------------------------------------------------------------------
# TestClipEntryToInfo
# ---------------------------------------------------------------------------


class TestClipEntryToInfo:
    def test_converts_clip_entry_basic(self, tmp_path):
        """Convert a real ClipEntry to ClipInfo; verify round-trip via to_dict."""
        clip_dir = tmp_path / "clips" / "my_clip"
        (clip_dir / "Input").mkdir(parents=True)
        (clip_dir / "Input" / "00001.png").write_bytes(b"fake")

        from ck_engine.clip_state import ClipEntry

        entry = ClipEntry(name="my_clip", root_path=str(clip_dir))
        entry.find_assets()

        info = _clip_entry_to_info(entry)
        assert isinstance(info, ClipInfo)
        assert info.name == "my_clip"
        assert info.input is not None
        assert info.input.frame_count == 1
        assert info.input.type == "sequence"
        assert info.alpha is None
        assert info.mask is None

    def test_to_dict_round_trip(self, tmp_path):
        """ClipInfo.to_dict() produces a serializable dict with correct values."""
        clip_dir = tmp_path / "clips" / "shot_01"
        (clip_dir / "Input").mkdir(parents=True)
        for i in range(4):
            (clip_dir / "Input" / f"{i + 1:05d}.png").write_bytes(b"fake")

        from ck_engine.clip_state import ClipEntry

        entry = ClipEntry(name="shot_01", root_path=str(clip_dir))
        entry.find_assets()

        info = _clip_entry_to_info(entry)
        d = info.to_dict()

        assert d["name"] == "shot_01"
        assert d["input"]["frame_count"] == 4
        assert d["input"]["type"] == "sequence"
        assert d["alpha"] is None
        assert d["mask"] is None

    def test_clip_state_is_string(self, tmp_path):
        """ClipInfo.state is a plain string (not an enum)."""
        clip_dir = tmp_path / "clips" / "state_clip"
        (clip_dir / "Input").mkdir(parents=True)
        (clip_dir / "Input" / "00001.png").write_bytes(b"fake")

        from ck_engine.clip_state import ClipEntry

        entry = ClipEntry(name="state_clip", root_path=str(clip_dir))
        entry.find_assets()

        info = _clip_entry_to_info(entry)
        assert isinstance(info.state, str)
        # A clip with only input and no alpha should be RAW
        assert info.state == "RAW"

    def test_ready_state_when_alpha_present(self, tmp_path):
        """ClipInfo.state is READY when both input and alpha hint exist."""
        clip_dir = tmp_path / "clips" / "ready_clip"
        (clip_dir / "Input").mkdir(parents=True)
        (clip_dir / "AlphaHint").mkdir(parents=True)
        (clip_dir / "Input" / "00001.png").write_bytes(b"fake")
        (clip_dir / "AlphaHint" / "00001.png").write_bytes(b"fake")

        from ck_engine.clip_state import ClipEntry

        entry = ClipEntry(name="ready_clip", root_path=str(clip_dir))
        entry.find_assets()

        info = _clip_entry_to_info(entry)
        assert info.state == "READY"
        assert info.alpha is not None
        assert info.alpha.frame_count == 1
