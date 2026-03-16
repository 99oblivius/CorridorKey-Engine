"""VideoMaMa AlphaGenerator wrapper."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from ck_engine.config import IMAGE_EXTS, VIDEO_EXTS, Dir

logger = logging.getLogger(__name__)


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _load_input_frames(input_dir: str) -> tuple[list[np.ndarray], list[str]]:
    """Load input frames as uint8 RGB, return (frames, stems)."""
    input_path = Path(input_dir)

    if _is_video(input_path):
        frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(str(input_path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        stem = input_path.stem
        stems = [f"{stem}_{i:05d}" for i in range(len(frames))]
        return frames, stems

    # Image sequence directory
    files = sorted(
        f for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in IMAGE_EXTS
    )
    frames = []
    stems = []
    for f in files:
        fpath = os.path.join(input_dir, f)
        if f.lower().endswith(".exr"):
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = np.clip(img, 0.0, 1.0)
                img = img ** (1.0 / 2.2)
                img = (img * 255.0).astype(np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.imread(fpath)

        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            stems.append(Path(f).stem)

    return frames, stems


def _load_mask_frames(mask_dir: str) -> list[np.ndarray]:
    """Load mask frames as uint8 grayscale with binary threshold."""
    mask_path = Path(mask_dir)
    masks: list[np.ndarray] = []

    if _is_video(mask_path):
        cap = cv2.VideoCapture(str(mask_path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                m = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                masks.append(m)
        finally:
            cap.release()
        return masks

    if mask_path.is_dir():
        files = sorted(
            f for f in os.listdir(mask_dir)
            if Path(f).suffix.lower() in IMAGE_EXTS
        )
        for f in files:
            fpath = os.path.join(mask_dir, f)
            if f.lower().endswith(".exr"):
                os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                m = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if m is not None:
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    m = np.clip(m, 0.0, 1.0)
                    m = (m * 255.0).astype(np.uint8)
            else:
                m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                masks.append(m)

    return masks


class VideoMaMaAlphaGenerator:
    """AlphaGenerator wrapper around VideoMaMa inference pipeline."""

    name = "videomama"
    is_temporal = True
    requires_mask = True

    def __init__(self, device: str = "cpu") -> None:
        from .inference import load_videomama_model

        self._pipeline = load_videomama_model(device=device)

    def generate(
        self,
        input_dir: str,
        output_dir: str,
        *,
        mask_dir: str | None = None,
        frame_indices: Sequence[int] | None = None,
        skip_existing: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        if mask_dir is None:
            raise ValueError(f"VideoMaMa requires mask_dir ({Dir.VIDEOMAMA_HINT})")

        os.makedirs(output_dir, exist_ok=True)

        # Load frames
        input_frames, stems = _load_input_frames(input_dir)
        mask_frames = _load_mask_frames(mask_dir)

        # Truncate to matching length
        num_frames = min(len(input_frames), len(mask_frames))
        input_frames = input_frames[:num_frames]
        mask_frames = mask_frames[:num_frames]
        stems = stems[:num_frames]

        if num_frames == 0:
            logger.error("No valid frame pairs found")
            return 0

        indices_set = set(frame_indices) if frame_indices is not None else None

        # Run inference
        from .inference import run_inference as run_videomama_frames

        total_written = 0
        frame_idx = 0
        for chunk_frames in run_videomama_frames(
            self._pipeline, input_frames, mask_frames
        ):
            for frame in chunk_frames:
                if frame_idx >= num_frames:
                    break

                # Write only selected indices
                if indices_set is None or frame_idx in indices_set:
                    name = stems[frame_idx]
                    out_path = os.path.join(output_dir, f"{name}.png")
                    # Extract channel 0 — model outputs identical RGB channels
                    frame_gray = frame[:, :, 0]
                    cv2.imwrite(out_path, frame_gray)
                    total_written += 1

                frame_idx += 1
                if on_progress:
                    on_progress(frame_idx, num_frames)

            logger.info(f"  Saved {total_written}/{num_frames} frames...")

        logger.info(f"VideoMaMa complete: {total_written} frames to {output_dir}")
        return total_written
