"""Alpha hint generation pipeline."""

from __future__ import annotations

import logging
import os
import shutil
from enum import Enum
from typing import TYPE_CHECKING

import ck_engine.pipeline as _pipeline_pkg

from ck_engine.clip_state import ClipEntry
from ck_engine.config import Dir
from ck_engine.project import is_video_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class AlphaMode(Enum):
    """Controls behaviour when AlphaHint/ already exists."""

    REPLACE = "replace"  # Delete existing, regenerate all
    FILL = "fill"  # Keep existing, only generate missing frames
    SKIP = "skip"  # Do nothing


def _discover_mask_dir(clip: ClipEntry) -> str | None:
    """Auto-discover VideoMamaMaskHint/ directory or video in a clip."""
    for f in os.listdir(clip.root_path):
        stem, _ = os.path.splitext(f)
        if stem.lower() == "videomamamaskhint":
            path = os.path.join(clip.root_path, f)
            if os.path.isdir(path) and os.listdir(path):
                return path
            if os.path.isfile(path) and is_video_file(path):
                return path
    return None


def generate_alpha_hints(
    clips: Sequence[ClipEntry],
    model: str,
    device: str | None = None,
    *,
    alpha_mode: AlphaMode = AlphaMode.REPLACE,
    start: int | None = None,
    end: int | None = None,
    mask_dir: str | None = None,
    on_clip_start: Callable[[str, int], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Generate alpha hints for clips using the specified model.

    Parameters
    ----------
    clips : list[ClipEntry]
        Clips to process.
    model : str
        Generator name: "gvm", "birefnet", or "videomama".
    device : str | None
        Compute device. None = auto-detect.
    alpha_mode : AlphaMode
        How to handle existing AlphaHint/ directory.
    start : int | None
        Start frame (0-based, inclusive).
    end : int | None
        End frame (0-based, inclusive).
    mask_dir : str | None
        Override mask directory (videomama only). Auto-discovers if None.
    on_clip_start : callback
        Called with (clip_name, total_clips) when starting a clip.
    on_progress : callback
        Called with (frames_done, total_frames).

    Returns
    -------
    int
        Number of clips that were processed successfully.

    Raises
    ------
    ImportError, RuntimeError
        If the requested model fails to initialize. Callers should treat this
        as a hard failure and surface a user-friendly error message.
    """
    if alpha_mode is AlphaMode.SKIP:
        return 0

    if device is None:
        device = _pipeline_pkg.resolve_device()

    from ck_engine.generators import get_generator

    # Let ImportError / RuntimeError propagate — the model is unusable and the
    # caller must decide how to report this to the user.
    generator = get_generator(model, device=device)

    succeeded: list[str] = []
    failed: list[str] = []

    for clip in clips:
        logger.info(f"Generating alpha ({model}) for: {clip.name}")
        if on_clip_start:
            on_clip_start(clip.name, len(clips))

        alpha_output_dir = os.path.join(clip.root_path, Dir.ALPHA_HINT)

        # Handle alpha_mode
        if alpha_mode is AlphaMode.REPLACE and os.path.exists(alpha_output_dir):
            shutil.rmtree(alpha_output_dir)
        skip_existing = alpha_mode is AlphaMode.FILL

        # Resolve frame_indices per clip
        clip_frame_indices = None
        if start is not None or end is not None:
            s = start if start is not None else 0
            e = end
            if e is not None:
                clip_frame_indices = range(s, e + 1)
            else:
                clip_frame_indices = range(s, clip.input_asset.frame_count)

        # Resolve mask_dir for videomama
        clip_mask_dir = mask_dir
        if generator.requires_mask and clip_mask_dir is None:
            clip_mask_dir = _discover_mask_dir(clip)
            if clip_mask_dir is None:
                logger.warning(f"Skipping {clip.name}: no VideoMamaMaskHint/ found")
                failed.append(clip.name)
                continue

        try:
            count = generator.generate(
                input_dir=clip.input_asset.path,
                output_dir=alpha_output_dir,
                mask_dir=clip_mask_dir,
                frame_indices=clip_frame_indices,
                skip_existing=skip_existing,
                on_progress=on_progress,
            )
            logger.info(f"Saved {count} alpha frames to {alpha_output_dir}")
            succeeded.append(clip.name)
        except Exception as e:
            logger.error(f"Error generating alpha ({model}) for {clip.name}: {e}")
            import traceback

            traceback.print_exc()
            failed.append(clip.name)

    if failed:
        logger.error(
            "Alpha generation failed for %d/%d clip(s): %s",
            len(failed),
            len(clips),
            ", ".join(failed),
        )

    return len(succeeded)
