"""Job execution for the CorridorKey engine.

The JobRunner bridges between the JSON-RPC API types and the existing
pipeline functions.  It scans clips, emits events, and delegates to
``generate_alpha_hints`` and ``run_inference`` from ``backend.pipeline``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from ck_engine.api.errors import INVALID_PATH, NO_VALID_CLIPS, EngineError


class _ThroughputTracker:
    """3-second EWMA throughput tracker for frames-per-second.

    Receives cumulative frame counts at irregular intervals and computes
    a smoothed fps that reflects actual sustained throughput — including
    batching, multi-GPU parallelism, and pipeline overlap.
    """

    def __init__(self, window: float = 3.0) -> None:
        self._window = window
        self._samples: list[tuple[float, int]] = []

    def record(self, now: float, done: int) -> float:
        """Record a sample and return the current smoothed fps."""
        self._samples.append((now, done))
        cutoff = now - self._window * 2
        self._samples = [(t, d) for t, d in self._samples if t >= cutoff]
        return self._fps()

    def reset(self) -> None:
        self._samples.clear()

    def _fps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        now = self._samples[-1][0]
        tau = self._window
        weighted_rate = 0.0
        total_weight = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            if dt <= 0:
                continue
            rate = (self._samples[i][1] - self._samples[i - 1][1]) / dt
            age = now - self._samples[i][0]
            weight = 2.0 ** (-age / tau)
            weighted_rate += rate * weight
            total_weight += weight
        return weighted_rate / total_weight if total_weight > 0 else 0.0
from ck_engine.api.events import (
    ClipCompleted,
    ClipStarted,
    JobAccepted,
    JobCancelled,
    JobCompleted,
    JobFailed,
    JobProgress,
)
from ck_engine.api.frames import parse_frame_range
from ck_engine.api.types import AssetInfo, ClipInfo, GenerateParams, InferenceParams

if TYPE_CHECKING:
    from ck_engine.engine.event_bus import EventBus
    from ck_engine.engine.model_pool import ModelPool

logger = logging.getLogger(__name__)


class JobRunner:
    """Executes generate and inference jobs."""

    def __init__(self, model_pool: ModelPool, event_bus: EventBus) -> None:
        self._model_pool = model_pool
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_project(self, path: str) -> dict:
        """Scan a project path and return clip info.

        Returns dict with keys: project_path, is_v2, clips (list of ClipInfo dicts)
        """
        if not os.path.isdir(path):
            raise EngineError(INVALID_PATH, f"Not a directory: {path}")

        from ck_engine.clip_state import ClipEntry
        from ck_engine.project import get_clip_dirs, is_v2_project

        clip_dirs = get_clip_dirs(path)
        clips: list[dict] = []

        for clip_dir in clip_dirs:
            try:
                name = os.path.basename(os.path.normpath(clip_dir))
                entry = ClipEntry(name=name, root_path=clip_dir)
                entry.find_assets()
                clips.append(_clip_entry_to_info(entry).to_dict())
            except Exception as exc:
                logger.warning("Failed to scan clip %s: %s", clip_dir, exc)

        return {
            "project_path": path,
            "is_v2": is_v2_project(path),
            "clips": clips,
        }

    def run_generate(
        self, job_id: str, params: GenerateParams, cancel: threading.Event
    ) -> None:
        """Execute an alpha generation job.  Runs in a worker thread."""
        try:
            # Scan clips
            clips = self._scan_clips(params.path)
            valid = [c for c in clips if c.input_asset and c.input_asset.frame_count > 0]
            if not valid:
                raise EngineError(NO_VALID_CLIPS, "No clips with input frames found")

            total_frames = sum(c.input_asset.frame_count for c in valid)
            self._event_bus.emit(JobAccepted(
                job_id=job_id, type="generate", total_frames=total_frames,
            ))

            # Parse frame range
            frame_start = None
            frame_end = None
            if params.frames:
                # For generate, we pass start/end to generate_alpha_hints
                # Parse the range to get the bounds
                indices = parse_frame_range(params.frames, max(c.input_asset.frame_count for c in valid))
                if indices:
                    frame_start = min(indices)
                    frame_end = max(indices)

            # Resolve device
            device = params.device if params.device != "auto" else None

            # Map alpha mode
            from ck_engine.pipeline import AlphaMode

            mode_map = {"replace": AlphaMode.REPLACE, "fill": AlphaMode.FILL, "skip": AlphaMode.SKIP}
            alpha_mode = mode_map.get(params.mode, AlphaMode.REPLACE)

            # Track progress per clip
            clips_ok = 0
            clips_failed = 0
            current_clip_name = ""
            t0 = time.monotonic()
            fps_tracker = _ThroughputTracker(window=3.0)

            def on_clip_start(clip_name: str, total_clips: int) -> None:
                nonlocal current_clip_name
                current_clip_name = clip_name
                fps_tracker.reset()
                clip_idx = next((i for i, c in enumerate(valid) if c.name == clip_name), 0)
                # total_clips is the number of clips, not frames — look up the frame count.
                clip_entry = valid[clip_idx] if clip_idx < len(valid) else None
                frame_count = clip_entry.input_asset.frame_count if clip_entry and clip_entry.input_asset else 0
                self._event_bus.emit(ClipStarted(
                    job_id=job_id, clip=clip_name, frames=frame_count,
                    clip_index=clip_idx, clips_total=len(valid),
                ))

            def on_progress(done: int, total: int) -> None:
                if cancel.is_set():
                    return
                fps = fps_tracker.record(time.monotonic(), done)
                self._event_bus.emit(JobProgress(
                    job_id=job_id, clip=current_clip_name,
                    done=done, total=total, fps=fps,
                ))

            from ck_engine.pipeline import generate_alpha_hints

            succeeded = generate_alpha_hints(
                valid,
                model=params.model,
                device=device,
                alpha_mode=alpha_mode,
                start=frame_start,
                end=frame_end,
                on_clip_start=on_clip_start,
                on_progress=on_progress,
            )

            clips_ok = succeeded
            clips_failed = len(valid) - succeeded
            elapsed = time.monotonic() - t0

            if cancel.is_set():
                self._event_bus.emit(JobCancelled(job_id=job_id, frames_completed=0))
                return

            self._event_bus.emit(JobCompleted(
                job_id=job_id,
                clips_ok=clips_ok,
                clips_failed=clips_failed,
                total_frames=total_frames,
                frames_ok=total_frames,  # generate doesn't track per-frame failures
                frames_failed=0,
                elapsed_seconds=elapsed,
            ))

        except EngineError:
            raise  # Let the caller handle API errors
        except Exception as exc:
            err_msg = str(exc) or f"{type(exc).__name__}: {exc!r}"
            self._event_bus.emit(JobFailed(job_id=job_id, error=err_msg))

    def run_inference(
        self, job_id: str, params: InferenceParams, cancel: threading.Event
    ) -> None:
        """Execute an inference job.  Runs in a worker thread."""
        try:
            # Scan clips -- need both input and alpha
            clips = self._scan_clips(params.path)
            ready = [c for c in clips
                     if c.input_asset and c.alpha_asset
                     and c.input_asset.frame_count > 0
                     and c.alpha_asset.frame_count > 0]
            if not ready:
                raise EngineError(NO_VALID_CLIPS, "No clips with both input and alpha frames")

            total_frames = sum(min(c.input_asset.frame_count, c.alpha_asset.frame_count) for c in ready)
            self._event_bus.emit(JobAccepted(
                job_id=job_id, type="inference", total_frames=total_frames,
            ))

            # Build InferenceSettings for the pipeline
            from ck_engine.pipeline import InferenceSettings as PipelineSettings

            settings = PipelineSettings(
                input_is_linear=params.settings.input_is_linear,
                despill_strength=params.settings.despill_strength,
                auto_despeckle=params.settings.auto_despeckle,
                despeckle_size=params.settings.despeckle_size,
                refiner_scale=params.settings.refiner_scale,
            )

            # Build OptimizationConfig if specified
            opt_config = None
            if params.optimization is not None:
                opt_config = self._build_optimization_config(params.optimization)

            # Resolve devices list
            devices = params.devices
            device = params.device if params.device != "auto" else None
            backend_name = params.backend if params.backend != "auto" else None

            # Parse frame range
            frame_start = None
            frame_end = None
            if params.frames:
                indices = parse_frame_range(params.frames, total_frames)
                if indices:
                    frame_start = min(indices)
                    frame_end = max(indices)

            # Progress tracking
            current_clip_name = ""
            t0 = time.monotonic()
            fps_tracker = _ThroughputTracker(window=3.0)

            def on_clip_start(clip_name: str, total: int) -> None:
                nonlocal current_clip_name
                current_clip_name = clip_name
                fps_tracker.reset()
                clip_idx = next((i for i, c in enumerate(ready) if c.name == clip_name), 0)
                self._event_bus.emit(ClipStarted(
                    job_id=job_id, clip=clip_name, frames=total,
                    clip_index=clip_idx, clips_total=len(ready),
                ))

            def on_progress(done: int, total: int, *args: int) -> None:
                if cancel.is_set():
                    return
                bytes_read = args[0] if len(args) > 0 else 0
                bytes_written = args[1] if len(args) > 1 else 0
                fps = fps_tracker.record(time.monotonic(), done)
                self._event_bus.emit(JobProgress(
                    job_id=job_id, clip=current_clip_name,
                    done=done, total=total,
                    bytes_read=bytes_read, bytes_written=bytes_written,
                    fps=fps,
                ))

            from ck_engine.pipeline import run_inference

            run_inference(
                ready,
                device=device,
                backend=backend_name,
                start=frame_start,
                end=frame_end,
                settings=settings,
                optimization_config=opt_config,
                devices=devices,
                img_size=params.img_size,
                read_workers=params.read_workers,
                write_workers=params.write_workers,
                cpus=params.cpus,
                gpu_resilience=params.gpu_resilience,
                on_clip_start=on_clip_start,
                on_progress=on_progress,
            )

            elapsed = time.monotonic() - t0

            if cancel.is_set():
                self._event_bus.emit(JobCancelled(job_id=job_id, frames_completed=0))
                return

            self._event_bus.emit(JobCompleted(
                job_id=job_id,
                clips_ok=len(ready),
                clips_failed=0,
                total_frames=total_frames,
                frames_ok=total_frames,
                frames_failed=0,
                elapsed_seconds=elapsed,
            ))

        except EngineError:
            raise
        except Exception as exc:
            err_msg = str(exc) or f"{type(exc).__name__}: {exc!r}"
            self._event_bus.emit(JobFailed(job_id=job_id, error=err_msg))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_clips(self, path: str) -> list:
        """Scan a path for clips, returning ClipEntry objects."""
        if not os.path.isdir(path):
            raise EngineError(INVALID_PATH, f"Not a directory: {path}")

        from ck_engine.clip_state import ClipEntry
        from ck_engine.project import get_clip_dirs

        clip_dirs = get_clip_dirs(path)
        if not clip_dirs:
            raise EngineError(NO_VALID_CLIPS, f"No clips found in: {path}")

        entries = []
        for clip_dir in clip_dirs:
            try:
                name = os.path.basename(os.path.normpath(clip_dir))
                entry = ClipEntry(name=name, root_path=clip_dir)
                entry.find_assets()
                entries.append(entry)
            except Exception as exc:
                logger.warning("Skipping clip %s: %s", clip_dir, exc)

        return entries

    @staticmethod
    def _build_optimization_config(opt_params):
        """Build an OptimizationConfig from API OptimizationParams."""
        from CorridorKeyModule.optimization_config import OptimizationConfig

        # Start from profile if specified
        if opt_params.profile:
            try:
                config = OptimizationConfig.from_profile(opt_params.profile)
            except (ValueError, KeyError):
                config = OptimizationConfig()
        else:
            config = OptimizationConfig()

        # Apply explicit overrides (non-None fields)
        resolved = opt_params.resolve()
        if resolved:
            # OptimizationConfig is frozen, so we need to create a new one
            import dataclasses

            current = dataclasses.asdict(config)
            # Map API field names to OptimizationConfig field names
            for key, value in resolved.items():
                if key in current:
                    current[key] = value
            # Handle precision alias mapping
            if "model_precision" in current:
                prec = current["model_precision"]
                if prec == "fp16":
                    current["model_precision"] = "float16"
                elif prec == "bf16":
                    current["model_precision"] = "bfloat16"
                elif prec == "fp32":
                    current["model_precision"] = "float32"
            config = OptimizationConfig(**{k: v for k, v in current.items()
                                           if k in {f.name for f in dataclasses.fields(OptimizationConfig)}})

        return config


def _clip_entry_to_info(entry) -> ClipInfo:
    """Convert a backend ClipEntry to an API ClipInfo."""

    def _asset(a) -> AssetInfo | None:
        if a is None:
            return None
        return AssetInfo(type=a.asset_type, frame_count=a.frame_count, path=a.path)

    return ClipInfo(
        name=entry.name,
        root_path=entry.root_path,
        state=entry.state.value if hasattr(entry.state, "value") else str(entry.state),
        input=_asset(entry.input_asset),
        alpha=_asset(entry.alpha_asset),
        mask=_asset(entry.mask_asset),
        has_outputs=entry.has_outputs,
        completed_frames=entry.completed_frame_count(),
    )
