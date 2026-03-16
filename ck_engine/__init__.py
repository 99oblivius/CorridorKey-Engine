"""ck_engine — engine layer for CorridorKey.

Imports are lazy to avoid pulling in torch/cv2 when only lightweight
submodules (e.g., ck_engine.path_utils, ck_engine.natural_sort) are needed.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Python 3.14 subprocess workaround (applied once at import time)
# ---------------------------------------------------------------------------
# Python 3.14 tightened FD validation in _posixsubprocess.fork_exec, raising
# "bad value(s) in fds_to_keep" when stale FDs are inherited.  Libraries like
# HuggingFace, multiprocessing, and tokenizers call fork_exec both through
# subprocess.Popen AND directly.  Patching at the _posixsubprocess level
# catches every caller.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.version_info >= (3, 14) and sys.platform != "win32":
    import _posixsubprocess

    _orig_fork_exec = _posixsubprocess.fork_exec

    def _safe_fork_exec(*args):  # type: ignore[no-untyped-def]
        try:
            return _orig_fork_exec(*args)
        except ValueError as exc:
            if "fds_to_keep" not in str(exc):
                raise
            # fds_to_keep is arg index 3 — filter out stale FDs and retry.
            args_list = list(args)
            fds_to_keep = args_list[3]
            args_list[3] = tuple(fd for fd in fds_to_keep if _fd_valid(fd))
            return _orig_fork_exec(*args_list)

    def _fd_valid(fd: int) -> bool:
        try:
            os.fstat(fd)
            return True
        except OSError:
            return False

    _posixsubprocess.fork_exec = _safe_fork_exec  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from .clip_state import (
        ClipAsset,
        ClipEntry,
        ClipState,
    )
    from .device import clear_device_cache, detect_best_device, resolve_device
    from .errors import CorridorKeyError
    from .model_manager import ModelManager
    from .natural_sort import natsorted, natural_sort_key
    from .pipeline import InferenceSettings
    from .project import (
        VIDEO_FILE_FILTER,
        add_clips_to_project,
        create_project,
        get_clip_dirs,
        is_image_file,
        is_v2_project,
        is_video_file,
        projects_root,
        sanitize_stem,
    )


def __getattr__(name: str) -> Any:
    """Lazy-load submodule attributes on first access."""
    _module_map = {
        # clip_state
        "ClipAsset": ".clip_state",
        "ClipEntry": ".clip_state",
        "ClipState": ".clip_state",
        # device
        "clear_device_cache": ".device",
        "detect_best_device": ".device",
        "resolve_device": ".device",
        # errors
        "CorridorKeyError": ".errors",
        # natural_sort
        "natsorted": ".natural_sort",
        "natural_sort_key": ".natural_sort",
        # pipeline
        "InferenceSettings": ".pipeline",
        # project
        "VIDEO_FILE_FILTER": ".project",
        "add_clips_to_project": ".project",
        "create_project": ".project",
        "get_clip_dirs": ".project",
        "is_image_file": ".project",
        "is_v2_project": ".project",
        "is_video_file": ".project",
        "projects_root": ".project",
        "sanitize_stem": ".project",
        # model_manager (heavy -- torch) — DEPRECATED, use ck_engine.engine.model_pool instead
        "ModelManager": ".model_manager",
    }

    if name in _module_map:
        module = importlib.import_module(_module_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VIDEO_FILE_FILTER",
    "ClipAsset",
    "ClipEntry",
    "ClipState",
    "CorridorKeyError",
    "InferenceSettings",
    "ModelManager",
    "add_clips_to_project",
    "clear_device_cache",
    "create_project",
    "detect_best_device",
    "get_clip_dirs",
    "is_image_file",
    "is_v2_project",
    "is_video_file",
    "natsorted",
    "natural_sort_key",
    "projects_root",
    "resolve_device",
    "sanitize_stem",
]
