"""Canonical defaults and constants for CorridorKey.

Every magic value that is referenced in more than one module lives here.
Submodules import from this file so that a single edit propagates everywhere.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Directory names (project structure)
# ---------------------------------------------------------------------------


class Dir(StrEnum):
    """Subdirectory names inside a clip folder."""

    INPUT = "Input"
    OUTPUT = "Output"
    ALPHA_HINT = "AlphaHint"
    VIDEOMAMA_HINT = "VideoMamaMaskHint"
    FG = "FG"
    MATTE = "Matte"
    COMP = "Comp"
    PROCESSED = "Processed"


# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------

IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".dpx"}
)
VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"}
)

# ---------------------------------------------------------------------------
# Model / pipeline numeric defaults
# ---------------------------------------------------------------------------

DEFAULT_IMG_SIZE: int = 2048
DEFAULT_TILE_SIZE: int = 512
DEFAULT_TILE_OVERLAP: int = 128

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------

DEFAULT_DESPILL_STRENGTH: float = 0.5  # 0.0–1.0
DEFAULT_DESPECKLE_SIZE: int = 400
DEFAULT_REFINER_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Postprocessing defaults (clean_matte / create_checkerboard)
# ---------------------------------------------------------------------------

DEFAULT_MATTE_DILATION: int = 25
DEFAULT_MATTE_BLUR: int = 5

DEFAULT_CHECKER_SIZE: int = 128
DEFAULT_CHECKER_COLOR1: float = 0.15
DEFAULT_CHECKER_COLOR2: float = 0.55

# ---------------------------------------------------------------------------
# ImageNet normalization
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# EXR output
# ---------------------------------------------------------------------------

# Deferred to avoid importing cv2 at config-load time.
# Use ``get_exr_write_flags()`` instead of a bare constant.
_exr_flags: list[int] | None = None


def get_exr_write_flags() -> list[int]:
    """Return OpenCV EXR write parameters (PXR24 half-float)."""
    global _exr_flags  # noqa: PLW0603
    if _exr_flags is None:
        import cv2

        _exr_flags = [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PXR24,
        ]
    return _exr_flags
