"""Comprehensive tests for the backend.api package.

Covers:
- backend.api.types  (TestTypes)
- backend.api.events (TestEvents)
- backend.api.errors (TestErrors)
- backend.api.frames (TestFrames)
"""

from __future__ import annotations

import json

import pytest

from ck_engine.api.errors import (
    CANCELLED,
    DEVICE_UNAVAILABLE,
    ENGINE_BUSY,
    INVALID_PARAMS,
    INVALID_PATH,
    INVALID_REQUEST,
    JOB_NOT_FOUND,
    METHOD_NOT_FOUND,
    MODEL_LOAD_FAILURE,
    NO_VALID_CLIPS,
    PARSE_ERROR,
    EngineError,
    error_response,
    success_response,
)
from ck_engine.api.events import (
    ALL_EVENT_TYPES,
    EVENT_CATEGORIES,
    ClipCompleted,
    ClipStarted,
    JobAccepted,
    JobCancelled,
    JobCompleted,
    JobFailed,
    JobProgress,
    LogEvent,
    ModelLoaded,
    ModelLoading,
    ModelRecompiling,
    ModelUnloaded,
    WarningEvent,
)
from ck_engine.api.frames import format_frame_range, parse_frame_range
from ck_engine.api.types import (
    AssetInfo,
    ClipInfo,
    DeviceInfo,
    EngineCapabilities,
    EngineStatus,
    FailedFrameInfo,
    GenerateParams,
    InferenceParams,
    InferenceSettings,
    JobResult,
    JobStatus,
    LoadedModelInfo,
    OptimizationParams,
    VRAMInfo,
)
from ck_engine.config import (
    DEFAULT_DESPECKLE_SIZE,
    DEFAULT_DESPILL_STRENGTH,
    DEFAULT_IMG_SIZE,
    DEFAULT_REFINER_SCALE,
)


# ---------------------------------------------------------------------------
# TestTypes
# ---------------------------------------------------------------------------


class TestTypes:
    def test_asset_info_round_trip(self):
        original = AssetInfo(type="sequence", frame_count=100, path="/clips/shot_a")
        d = original.to_dict()
        restored = AssetInfo.from_dict(d)
        assert restored == original
        assert d == {"type": "sequence", "frame_count": 100, "path": "/clips/shot_a"}

    def test_clip_info_round_trip(self):
        asset = AssetInfo(type="video", frame_count=50, path="/clips/shot_b/input.mp4")
        original = ClipInfo(
            name="shot_b",
            root_path="/clips/shot_b",
            state="RAW",
            input=asset,
            alpha=None,
            mask=None,
            has_outputs=False,
            completed_frames=0,
        )
        d = original.to_dict()
        restored = ClipInfo.from_dict(d)
        assert restored == original
        # None fields are preserved in dict
        assert d["alpha"] is None
        assert d["mask"] is None
        # Nested AssetInfo round-trips
        assert d["input"]["type"] == "video"

    def test_clip_info_with_all_assets(self):
        input_asset = AssetInfo(type="sequence", frame_count=24, path="/c/Input")
        alpha_asset = AssetInfo(type="sequence", frame_count=24, path="/c/AlphaHint")
        mask_asset = AssetInfo(type="sequence", frame_count=24, path="/c/Mask")
        original = ClipInfo(
            name="shot_c",
            root_path="/c",
            state="READY",
            input=input_asset,
            alpha=alpha_asset,
            mask=mask_asset,
            has_outputs=True,
            completed_frames=12,
        )
        restored = ClipInfo.from_dict(original.to_dict())
        assert restored == original
        assert restored.alpha == alpha_asset
        assert restored.mask == mask_asset

    def test_inference_settings_defaults(self):
        s = InferenceSettings()
        assert s.input_is_linear is False
        assert s.despill_strength == DEFAULT_DESPILL_STRENGTH
        assert s.auto_despeckle is True
        assert s.despeckle_size == DEFAULT_DESPECKLE_SIZE
        assert s.refiner_scale == DEFAULT_REFINER_SCALE

    def test_inference_settings_round_trip(self):
        original = InferenceSettings(
            input_is_linear=True,
            despill_strength=0.8,
            auto_despeckle=False,
            despeckle_size=200,
            refiner_scale=0.5,
        )
        restored = InferenceSettings.from_dict(original.to_dict())
        assert restored == original

    def test_optimization_params_resolve_profile(self):
        p = OptimizationParams(profile="optimized")
        resolved = p.resolve()
        assert resolved["flash_attention"] is True
        assert resolved["tiled_refiner"] is True
        assert resolved["mixed_precision"] is True
        assert resolved["model_precision"] == "float16"
        assert resolved["cache_clearing"] is True
        assert resolved["disable_cudnn_benchmark"] is True
        # profile key itself should not appear in the output
        assert "profile" not in resolved

    def test_optimization_params_resolve_overrides(self):
        # Start from "optimized" profile but override specific fields
        p = OptimizationParams(profile="optimized", flash_attention=False, dma_buffers=4)
        resolved = p.resolve()
        # Override takes effect
        assert resolved["flash_attention"] is False
        assert resolved["dma_buffers"] == 4
        # Non-overridden profile values remain
        assert resolved["mixed_precision"] is True
        assert resolved["model_precision"] == "float16"

    def test_optimization_params_resolve_no_profile(self):
        p = OptimizationParams(flash_attention=True, model_precision="float32")
        resolved = p.resolve()
        assert resolved == {"flash_attention": True, "model_precision": "float32"}
        assert "profile" not in resolved
        # None fields must not appear
        assert "tiled_refiner" not in resolved

    def test_optimization_params_resolve_unknown_profile(self):
        p = OptimizationParams(profile="nonexistent")
        with pytest.raises(ValueError, match="Unknown optimization profile"):
            p.resolve()

    def test_optimization_params_round_trip(self):
        original = OptimizationParams(
            profile="performance",
            flash_attention=True,
            tile_size=256,
            dma_buffers=3,
        )
        restored = OptimizationParams.from_dict(original.to_dict())
        assert restored == original

    def test_generate_params_round_trip(self):
        original = GenerateParams(
            path="/clips/shot_a",
            model="birefnet",
            mode="replace",
            frames="1-50",
            device="cuda:0",
            halt_on_failure=True,
        )
        restored = GenerateParams.from_dict(original.to_dict())
        assert restored == original

    def test_generate_params_defaults(self):
        p = GenerateParams(path="/tmp")
        assert p.model == "birefnet"
        assert p.mode == "replace"
        assert p.frames is None
        assert p.device == "auto"
        assert p.halt_on_failure is False

    def test_inference_params_round_trip(self):
        settings = InferenceSettings(despill_strength=0.3)
        opt = OptimizationParams(profile="optimized", tile_size=512)
        original = InferenceParams(
            path="/clips/shot_a",
            frames="1-100",
            device="cuda:0",
            backend="torch",
            settings=settings,
            optimization=opt,
            devices=["cuda:0", "cuda:1"],
            img_size=1024,
            read_workers=2,
            write_workers=2,
            cpus=4,
            gpu_resilience=True,
            halt_on_failure=True,
        )
        restored = InferenceParams.from_dict(original.to_dict())
        assert restored == original

    def test_inference_params_defaults(self):
        p = InferenceParams(path="/tmp")
        assert p.frames is None
        assert p.device == "auto"
        assert p.backend == "auto"
        assert p.settings == InferenceSettings()
        assert p.optimization is None
        assert p.devices is None
        assert p.img_size == DEFAULT_IMG_SIZE
        assert p.read_workers == 0
        assert p.write_workers == 0
        assert p.cpus == 0
        assert p.gpu_resilience is False
        assert p.halt_on_failure is False

    def test_engine_capabilities_round_trip(self):
        devices = [
            DeviceInfo(id="cuda:0", name="RTX 4090", vram_gb=24.0),
            DeviceInfo(id="cuda:1", name="RTX 3080", vram_gb=10.0),
        ]
        original = EngineCapabilities(
            version="1.2.3",
            generators=["birefnet", "gvm"],
            backends=["torch", "onnx"],
            devices=devices,
            profiles=["original", "optimized"],
            transport="stdio",
        )
        restored = EngineCapabilities.from_dict(original.to_dict())
        assert restored == original
        assert len(restored.devices) == 2
        assert restored.devices[0].id == "cuda:0"
        assert restored.devices[1].vram_gb == 10.0

    def test_engine_status_round_trip(self):
        model_info = LoadedModelInfo(
            backend="torch",
            device="cuda:0",
            vram_mb=4096.0,
            config_hash="abc123",
            img_size=2048,
            precision="float16",
        )
        vram = VRAMInfo(total_mb=24576.0, used_mb=4096.0, free_mb=20480.0)
        original = EngineStatus(
            state="busy",
            active_job="job-42",
            models_loaded={"torch": model_info},
            vram=vram,
            uptime_seconds=3600.0,
        )
        restored = EngineStatus.from_dict(original.to_dict())
        assert restored == original
        assert restored.models_loaded["torch"].precision == "float16"
        assert restored.vram.free_mb == 20480.0

    def test_engine_status_idle_defaults(self):
        s = EngineStatus(state="idle")
        assert s.active_job is None
        assert s.models_loaded == {}
        assert s.vram is None
        assert s.uptime_seconds == 0.0

    def test_job_status_round_trip(self):
        original = JobStatus(
            job_id="job-99",
            state="running",
            type="inference",
            current_clip="shot_a",
            progress={"done": 25, "total": 100},
            clips_completed=1,
            clips_total=5,
            elapsed_seconds=12.5,
        )
        restored = JobStatus.from_dict(original.to_dict())
        assert restored == original

    def test_job_result_round_trip(self):
        clip1 = ClipInfo(
            name="shot_a",
            root_path="/clips/shot_a",
            state="RAW",
            input=AssetInfo(type="sequence", frame_count=50, path="/clips/shot_a/Input"),
        )
        clip2 = ClipInfo(
            name="shot_b",
            root_path="/clips/shot_b",
            state="READY",
        )
        original = JobResult(
            job_id="job-1",
            clips=[clip1, clip2],
            total_frames=75,
        )
        restored = JobResult.from_dict(original.to_dict())
        assert restored == original
        assert len(restored.clips) == 2
        assert restored.clips[0].name == "shot_a"

    def test_failed_frame_info_round_trip(self):
        original = FailedFrameInfo(clip="shot_a", frame=42, error="CUDA out of memory")
        restored = FailedFrameInfo.from_dict(original.to_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# TestEvents
# ---------------------------------------------------------------------------


class TestEvents:
    def test_all_events_have_method(self):
        for cls in ALL_EVENT_TYPES:
            assert hasattr(cls, "method"), f"{cls.__name__} missing 'method'"
            assert isinstance(cls.method, str), f"{cls.__name__}.method must be a str"
            assert cls.method, f"{cls.__name__}.method must not be empty"

    def test_all_events_have_category(self):
        valid_categories = EVENT_CATEGORIES - {"all"}
        for cls in ALL_EVENT_TYPES:
            assert hasattr(cls, "category"), f"{cls.__name__} missing 'category'"
            assert cls.category in valid_categories, (
                f"{cls.__name__}.category={cls.category!r} not in {valid_categories}"
            )

    def test_job_accepted_notification(self):
        event = JobAccepted(job_id="j1", type="inference", total_frames=200)
        note = event.to_notification()
        # Must be valid JSON
        json.dumps(note)
        assert note["jsonrpc"] == "2.0"
        assert note["method"] == "event.job.accepted"
        assert "id" not in note
        params = note["params"]
        assert params["job_id"] == "j1"
        assert params["type"] == "inference"
        assert params["total_frames"] == 200

    def test_job_progress_notification(self):
        event = JobProgress(
            job_id="j2",
            clip="shot_a",
            done=10,
            total=100,
            bytes_read=1024,
            bytes_written=512,
            fps=24.5,
        )
        note = event.to_notification()
        assert note["method"] == "event.job.progress"
        p = note["params"]
        assert p["done"] == 10
        assert p["total"] == 100
        assert p["fps"] == 24.5
        assert p["bytes_read"] == 1024

    def test_job_completed_notification(self):
        failed = [{"clip": "shot_a", "frame": 5, "error": "oom"}]
        event = JobCompleted(
            job_id="j3",
            clips_ok=3,
            clips_failed=1,
            total_frames=150,
            frames_ok=145,
            frames_failed=5,
            elapsed_seconds=60.0,
            failed_frames=failed,
        )
        note = event.to_notification()
        assert note["method"] == "event.job.completed"
        p = note["params"]
        assert p["clips_ok"] == 3
        assert p["frames_failed"] == 5
        assert p["failed_frames"] == failed

    def test_job_completed_no_failures(self):
        event = JobCompleted(
            job_id="j4",
            clips_ok=5,
            clips_failed=0,
            total_frames=100,
            frames_ok=100,
            frames_failed=0,
            elapsed_seconds=30.0,
        )
        note = event.to_notification()
        assert note["params"]["failed_frames"] is None

    def test_model_loaded_notification(self):
        event = ModelLoaded(
            model="birefnet",
            device="cuda:0",
            vram_mb=3500.0,
            load_seconds=4.2,
        )
        note = event.to_notification()
        assert note["method"] == "event.model.loaded"
        p = note["params"]
        assert p["model"] == "birefnet"
        assert p["vram_mb"] == 3500.0
        assert p["load_seconds"] == 4.2

    def test_log_event_has_timestamp(self):
        event = LogEvent(level="INFO", message="engine started")
        # timestamp should be auto-populated (close to current time)
        assert event.timestamp > 0
        note = event.to_notification()
        assert note["method"] == "event.log"
        assert note["params"]["timestamp"] > 0

    def test_log_event_custom_timestamp(self):
        event = LogEvent(level="WARNING", message="low VRAM", logger="engine", timestamp=1234567890.0)
        assert event.timestamp == 1234567890.0

    def test_warning_event_notification(self):
        event = WarningEvent(
            message="high memory pressure",
            clip="shot_a",
            detail="VRAM below 500 MB",
        )
        note = event.to_notification()
        assert note["method"] == "event.warning"
        p = note["params"]
        assert p["message"] == "high memory pressure"
        assert p["clip"] == "shot_a"
        assert p["detail"] == "VRAM below 500 MB"

    def test_event_notification_structure(self):
        """All events produce notifications with exactly jsonrpc/method/params and no id."""
        test_events = [
            JobAccepted(job_id="j", type="generate", total_frames=1),
            ClipStarted(job_id="j", clip="c", frames=10, clip_index=0, clips_total=1),
            JobProgress(job_id="j", clip="c", done=1, total=10),
            ClipCompleted(job_id="j", clip="c", frames_ok=10, frames_failed=0),
            JobCompleted(
                job_id="j", clips_ok=1, clips_failed=0, total_frames=10,
                frames_ok=10, frames_failed=0, elapsed_seconds=1.0
            ),
            JobFailed(job_id="j", error="crash"),
            JobCancelled(job_id="j", frames_completed=5),
            ModelLoading(model="birefnet", device="cuda:0"),
            ModelLoaded(model="birefnet", device="cuda:0", vram_mb=1000.0, load_seconds=1.0),
            ModelUnloaded(model="birefnet", freed_mb=1000.0),
            ModelRecompiling(reason="shape change", backend="torch"),
            LogEvent(level="INFO", message="test"),
            WarningEvent(message="test warning"),
        ]
        for event in test_events:
            note = event.to_notification()
            # Must be JSON-serializable
            json.dumps(note)
            assert "jsonrpc" in note, f"{type(event).__name__} missing jsonrpc"
            assert "method" in note, f"{type(event).__name__} missing method"
            assert "params" in note, f"{type(event).__name__} missing params"
            assert "id" not in note, f"{type(event).__name__} must not have id (not a request)"
            assert note["jsonrpc"] == "2.0"
            assert isinstance(note["params"], dict)

    def test_event_categories_contains_all(self):
        assert "all" in EVENT_CATEGORIES

    def test_all_event_types_count(self):
        # Sanity check: ALL_EVENT_TYPES has 13 entries as defined
        assert len(ALL_EVENT_TYPES) == 13


# ---------------------------------------------------------------------------
# TestErrors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_error_codes_unique(self):
        codes = [
            PARSE_ERROR,
            INVALID_REQUEST,
            METHOD_NOT_FOUND,
            INVALID_PARAMS,
            ENGINE_BUSY,
            JOB_NOT_FOUND,
            INVALID_PATH,
            NO_VALID_CLIPS,
            MODEL_LOAD_FAILURE,
            DEVICE_UNAVAILABLE,
            CANCELLED,
        ]
        assert len(codes) == len(set(codes)), "Error codes must all be unique"

    def test_engine_error_to_response(self):
        err = EngineError(code=ENGINE_BUSY, message="Engine is busy")
        response = err.to_response(request_id=42)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert "error" in response
        assert response["error"]["code"] == ENGINE_BUSY
        assert response["error"]["message"] == "Engine is busy"
        assert "data" not in response["error"]

    def test_engine_error_with_data(self):
        err = EngineError(code=INVALID_PARAMS, message="Bad params", data={"field": "path"})
        response = err.to_response(request_id="req-1")
        assert response["id"] == "req-1"
        assert response["error"]["data"] == {"field": "path"}

    def test_engine_error_with_none_id(self):
        err = EngineError(code=PARSE_ERROR, message="Cannot parse")
        response = err.to_response(request_id=None)
        assert response["id"] is None
        assert response["error"]["code"] == PARSE_ERROR

    def test_engine_error_is_exception(self):
        err = EngineError(code=JOB_NOT_FOUND, message="Job not found")
        with pytest.raises(EngineError):
            raise err

    def test_error_response_helper(self):
        response = error_response(
            request_id=7,
            code=MODEL_LOAD_FAILURE,
            message="Failed to load model",
            data={"model": "birefnet"},
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 7
        assert response["error"]["code"] == MODEL_LOAD_FAILURE
        assert response["error"]["message"] == "Failed to load model"
        assert response["error"]["data"] == {"model": "birefnet"}

    def test_success_response_helper(self):
        result = {"state": "idle", "version": "1.0"}
        response = success_response(request_id=3, result=result)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert response["result"] == result
        assert "error" not in response

    def test_error_response_no_data(self):
        response = error_response(request_id=1, code=INVALID_PATH, message="Path not found")
        assert "data" not in response["error"]

    def test_success_response_none_result(self):
        response = success_response(request_id=5, result=None)
        assert response["result"] is None

    def test_error_response_string_id(self):
        response = error_response(request_id="abc", code=CANCELLED, message="Cancelled")
        assert response["id"] == "abc"

    def test_response_is_json_serializable(self):
        err = EngineError(code=ENGINE_BUSY, message="busy", data={"queue": 3})
        response = err.to_response(request_id=1)
        # Should not raise
        json.dumps(response)

        ok_response = success_response(request_id=2, result={"clips": ["a", "b"]})
        json.dumps(ok_response)


# ---------------------------------------------------------------------------
# TestFrames
# ---------------------------------------------------------------------------


class TestFrames:
    # --- parse_frame_range ---

    def test_parse_none_returns_all(self):
        result = parse_frame_range(None, 10)
        assert result == list(range(10))

    def test_parse_empty_string_returns_all(self):
        result = parse_frame_range("", 10)
        assert result == list(range(10))

    def test_parse_single_frame(self):
        # "5" (1-based) -> index 4 (0-based)
        assert parse_frame_range("5", 10) == [4]

    def test_parse_range(self):
        # "1-10" (1-based inclusive) -> [0..9]
        assert parse_frame_range("1-10", 10) == list(range(10))

    def test_parse_open_range(self):
        # "5-" with total=10 -> indices [4..9]
        assert parse_frame_range("5-", 10) == list(range(4, 10))

    def test_parse_mixed(self):
        # "1,5,10-20" with total=100 -> [0, 4, 9..19]
        result = parse_frame_range("1,5,10-20", 100)
        expected = [0, 4] + list(range(9, 20))
        assert result == expected

    def test_parse_clamps_to_total(self):
        # "1-1000" with total=5 -> indices [0..4]
        result = parse_frame_range("1-1000", 5)
        assert result == list(range(5))

    def test_parse_deduplicates(self):
        # "1-5,3-7" has overlap at 3-5; result should be [0..6]
        result = parse_frame_range("1-5,3-7", 20)
        assert result == list(range(7))

    def test_parse_invalid_negative_raises(self):
        with pytest.raises(ValueError):
            parse_frame_range("-1", 10)

    def test_parse_invalid_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_frame_range("abc", 10)

    def test_parse_invalid_reversed_range_raises(self):
        # end < start
        with pytest.raises(ValueError, match="less than start"):
            parse_frame_range("10-5", 20)

    def test_parse_zero_frame_number_raises(self):
        # Frame numbers are 1-based; 0 is invalid
        with pytest.raises(ValueError):
            parse_frame_range("0", 10)

    def test_parse_zero_total_raises(self):
        with pytest.raises(ValueError, match="total must be > 0"):
            parse_frame_range(None, 0)

    def test_parse_negative_total_raises(self):
        with pytest.raises(ValueError, match="total must be > 0"):
            parse_frame_range("1-5", -1)

    def test_parse_single_frame_first(self):
        assert parse_frame_range("1", 5) == [0]

    def test_parse_single_frame_last(self):
        assert parse_frame_range("5", 5) == [4]

    def test_parse_single_clamps_beyond_total(self):
        # Frame 100 with total=5 clamps to last (index 4)
        result = parse_frame_range("100", 5)
        assert result == [4]

    def test_parse_open_range_from_first(self):
        assert parse_frame_range("1-", 5) == list(range(5))

    def test_parse_comma_separated_singles(self):
        assert parse_frame_range("1,3,5", 10) == [0, 2, 4]

    def test_parse_whitespace_tolerance(self):
        # Leading/trailing spaces in parts should be stripped
        result = parse_frame_range("1, 3, 5", 10)
        assert result == [0, 2, 4]

    def test_parse_two_two(self):
        # "2-2" -> single frame at index 1
        assert parse_frame_range("2-2", 5) == [1]

    # --- format_frame_range ---

    def test_format_consecutive(self):
        # [0,1,2,4,9,10,11] -> "1-3,5,10-12"
        assert format_frame_range([0, 1, 2, 4, 9, 10, 11]) == "1-3,5,10-12"

    def test_format_singles(self):
        assert format_frame_range([0, 4]) == "1,5"

    def test_format_empty(self):
        assert format_frame_range([]) == ""

    def test_format_single_element(self):
        assert format_frame_range([0]) == "1"

    def test_format_all_consecutive(self):
        assert format_frame_range([0, 1, 2, 3, 4]) == "1-5"

    def test_format_deduplicates_input(self):
        # Duplicates in input should be handled
        assert format_frame_range([0, 0, 1, 1, 2]) == "1-3"

    def test_format_unsorted_input(self):
        # Input need not be sorted
        assert format_frame_range([4, 0, 2, 1, 3]) == "1-5"

    # --- round_trip ---

    def test_round_trip(self):
        # parse -> format -> parse again yields same result
        original_spec = "1-3,5,10-12"
        total = 100
        indices_1 = parse_frame_range(original_spec, total)
        formatted = format_frame_range(indices_1)
        indices_2 = parse_frame_range(formatted, total)
        assert indices_1 == indices_2

    def test_round_trip_with_open_range(self):
        total = 20
        # Open range "5-" becomes "5-20" after format
        indices_1 = parse_frame_range("5-", total)
        formatted = format_frame_range(indices_1)
        indices_2 = parse_frame_range(formatted, total)
        assert indices_1 == indices_2

    def test_round_trip_mixed(self):
        total = 50
        indices_1 = parse_frame_range("1,5,10-20,45-", total)
        formatted = format_frame_range(indices_1)
        indices_2 = parse_frame_range(formatted, total)
        assert indices_1 == indices_2
