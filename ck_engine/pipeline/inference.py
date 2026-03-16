"""Inference pipeline for CorridorKey."""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
import shutil
from typing import TYPE_CHECKING

from ck_engine.clip_state import ClipEntry
from ck_engine.config import DEFAULT_DESPECKLE_SIZE, DEFAULT_DESPILL_STRENGTH, DEFAULT_IMG_SIZE, DEFAULT_REFINER_SCALE
from ck_engine.device import resolve_device
from ck_engine.natural_sort import natsorted
from ck_engine.project import is_image_file, is_video_file
from ck_engine.validators import ensure_output_dirs, validate_frame_counts

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferenceSettings:
    """Settings for CorridorKey inference.

    Can be constructed directly for non-interactive use (Nuke, Houdini, batch scripts).
    The single canonical settings class — used by CLI, service, and async pipeline.
    """

    input_is_linear: bool = False
    despill_strength: float = DEFAULT_DESPILL_STRENGTH  # 0.0-1.0 (canonical default)
    auto_despeckle: bool = True
    despeckle_size: int = DEFAULT_DESPECKLE_SIZE
    refiner_scale: float = DEFAULT_REFINER_SCALE

    def to_dict(self) -> dict:
        """Serialize to plain dict (for manifests / JSON persistence)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InferenceSettings:
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------


def _extract_video_frames(video_path: str, output_dir: str, max_frames: int | None = None) -> list[str]:
    """Extract video frames to PNGs using ffmpeg."""
    from ck_engine.ffmpeg_tools import extract_frames

    os.makedirs(output_dir, exist_ok=True)
    extract_frames(video_path, output_dir, pattern="%05d.png", total_frames=max_frames or 0)

    paths = natsorted(glob.glob(os.path.join(output_dir, "*.png")))
    if max_frames is not None:
        paths = paths[:max_frames]
    return paths


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    clips: Sequence[ClipEntry],
    device: str | None = None,
    backend: str | None = None,
    start: int | None = None,
    end: int | None = None,
    settings: InferenceSettings | None = None,
    optimization_config: object | None = None,
    devices: list[str] | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    read_workers: int = 0,
    write_workers: int = 0,
    cpus: int = 0,
    gpu_resilience: bool = False,
    *,
    on_clip_start: Callable[[str, int], None] | None = None,
    on_progress: Callable[[int, int, int, int], None] | None = None,
) -> None:
    ready_clips = [c for c in clips if c.input_asset and c.alpha_asset]

    if not ready_clips:
        logger.info("No clips found with both Input and Alpha assets. Run generate_coarse_alpha first?")
        return

    logger.info(f"Found {len(ready_clips)} clips ready for inference.")

    if settings is None:
        settings = InferenceSettings()

    logger.info(
        "Inference settings: linear=%s, despill=%.1f, despeckle=%s (size=%d), refiner=%.1f",
        settings.input_is_linear,
        settings.despill_strength,
        settings.auto_despeckle,
        settings.despeckle_size,
        settings.refiner_scale,
    )

    from ck_engine.async_pipeline import AsyncInferencePipeline, PipelineConfig

    config = PipelineConfig(
        img_size=img_size,
        backend=backend,
        devices=devices,
        optimization_config=optimization_config,
        read_workers=read_workers,
        write_workers=write_workers,
        cpus=cpus,
        comp_format=getattr(optimization_config, "comp_format", "exr") if optimization_config else "exr",
        gpu_resilience=gpu_resilience,
    )
    pipeline = AsyncInferencePipeline(config)
    pipeline.load_engines()

    import tempfile

    for clip in ready_clips:
        logger.info(f"Running Inference on: {clip.name}")

        dirs = ensure_output_dirs(clip.root_path)
        fg_dir = dirs["fg"]
        matte_dir = dirs["matte"]
        comp_dir = dirs["comp"]
        proc_dir = dirs["processed"]

        total_frames = validate_frame_counts(
            clip.name, clip.input_asset.frame_count, clip.alpha_asset.frame_count
        )
        frame_start = start or 0
        frame_end = end if end is not None else total_frames - 1
        frame_end = min(frame_end, total_frames - 1)
        num_frames = frame_end - frame_start + 1
        logger.info(
            f"  Input frames: {clip.input_asset.frame_count},"
            f" Alpha frames: {clip.alpha_asset.frame_count}"
            f" -> Processing frames {frame_start}-{frame_start + num_frames - 1}"
            f" ({num_frames} frames)"
        )

        if num_frames <= 0:
            logger.warning(f"Clip '{clip.name}': 0 frames to process, skipping.")
            continue

        if on_clip_start:
            on_clip_start(clip.name, num_frames)

        tmp_dirs = []
        try:
            if clip.input_asset.asset_type == "video":
                tmp_input = tempfile.mkdtemp(prefix="ck_input_")
                tmp_dirs.append(tmp_input)
                logger.info("  Extracting video frames for parallel processing...")
                extracted = _extract_video_frames(clip.input_asset.path, tmp_input, max_frames=frame_start + num_frames)
                input_paths = extracted[frame_start:]
                input_stems = [f"{i:05d}" for i in range(frame_start, frame_start + len(input_paths))]
            else:
                input_files = natsorted([f for f in os.listdir(clip.input_asset.path) if is_image_file(f)])
                input_files = input_files[frame_start:frame_start + num_frames]
                input_paths = [os.path.join(clip.input_asset.path, f) for f in input_files]
                input_stems = [os.path.splitext(f)[0] for f in input_files]

            if clip.alpha_asset.asset_type == "video":
                tmp_alpha = tempfile.mkdtemp(prefix="ck_alpha_")
                tmp_dirs.append(tmp_alpha)
                logger.info("  Extracting alpha video frames...")
                alpha_extracted = _extract_video_frames(clip.alpha_asset.path, tmp_alpha, max_frames=frame_start + num_frames)
                alpha_paths = alpha_extracted[frame_start:]
            else:
                alpha_files = natsorted([f for f in os.listdir(clip.alpha_asset.path) if is_image_file(f)])
                alpha_paths = [os.path.join(clip.alpha_asset.path, f) for f in alpha_files[frame_start:frame_start + num_frames]]

            result = pipeline.process_clip(
                input_paths=input_paths,
                alpha_paths=alpha_paths,
                input_stems=input_stems,
                output_dirs={"fg": fg_dir, "matte": matte_dir, "comp": comp_dir, "processed": proc_dir},
                settings=settings,
                on_progress=on_progress,
            )

            logger.info(
                f"Clip {clip.name} Complete: {result['completed']}/{result['total']} frames ({result['failed']} failed)"
            )

        finally:
            for tmp in tmp_dirs:
                shutil.rmtree(tmp, ignore_errors=True)
