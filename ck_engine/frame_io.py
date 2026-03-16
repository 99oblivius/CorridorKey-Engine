"""Unified frame I/O — read images and video frames as float32 RGB.

All reading functions return float32 arrays in [0, 1] range with RGB channel
order. EXR files are read as-is (linear float); standard formats (PNG, JPG,
etc.) are normalized from uint8.

This module consolidates frame-reading patterns that were previously duplicated
across service.py methods (_read_input_frame, reprocess_single_frame,
_load_frames_for_videomama, _load_mask_frames_for_videomama).
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

from .validators import normalize_mask_channels, normalize_mask_dtype

logger = logging.getLogger(__name__)


def read_image_frame(fpath: str, gamma_correct_exr: bool = False) -> np.ndarray | None:
    """Read an image file (EXR or standard) as float32 RGB [0, 1].

    Args:
        fpath: Absolute path to image file.
        gamma_correct_exr: If True, apply gamma 1/2.2 to EXR data
            (converts linear → approximate sRGB for models expecting sRGB).

    Returns:
        float32 array [H, W, 3] in RGB order, or None if read fails.
    """
    is_exr = fpath.lower().endswith(".exr")

    if is_exr:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read frame: %s", fpath)
            return None
        # Strip alpha channel from BGRA EXR
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = np.maximum(img_rgb, 0.0).astype(np.float32)
        if gamma_correct_exr:
            result = np.power(result, 1.0 / 2.2).astype(np.float32)
        return result
    img = cv2.imread(fpath)
    if img is None:
        logger.warning("Could not read frame: %s", fpath)
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def read_mask_frame(fpath: str, clip_name: str = "", frame_index: int = 0) -> np.ndarray | None:
    """Read a mask frame as float32 [H, W] in [0, 1].

    Handles any channel count and dtype via normalize_mask_channels/dtype.

    Args:
        fpath: Path to mask image.
        clip_name: For error context in normalization.
        frame_index: For error context in normalization.

    Returns:
        float32 array [H, W] in [0, 1], or None if read fails.
    """
    mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if mask_in is None:
        return None
    # dtype normalization MUST happen before channel extraction, because
    # normalize_mask_channels casts to float32 — which would make a uint8
    # 255 into float32 255.0, skipping the /255 division in normalize_mask_dtype.
    mask = normalize_mask_dtype(mask_in)
    mask = normalize_mask_channels(mask, clip_name, frame_index)
    return mask
