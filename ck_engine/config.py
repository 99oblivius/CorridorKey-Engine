"""Canonical defaults and constants for CorridorKey.

Re-exports everything from :mod:`CorridorKeyModule.constants` so that
existing ``from ck_engine.config import X`` statements continue to work.
The authoritative definitions live in ``CorridorKeyModule/constants.py``
to avoid a circular dependency between ``CorridorKeyModule/`` and ``backend/``.
"""

from __future__ import annotations

from CorridorKeyModule.constants import (  # noqa: F401
    DEFAULT_CHECKER_COLOR1,
    DEFAULT_CHECKER_COLOR2,
    DEFAULT_CHECKER_SIZE,
    DEFAULT_DESPECKLE_SIZE,
    DEFAULT_DESPILL_STRENGTH,
    DEFAULT_IMG_SIZE,
    DEFAULT_MATTE_BLUR,
    DEFAULT_MATTE_DILATION,
    DEFAULT_REFINER_SCALE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    Dir,
    IMAGE_EXTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIDEO_EXTS,
    get_exr_write_flags,
)
