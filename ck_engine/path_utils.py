"""Path constants and Windows→Linux path mapping for the CLI pipeline."""

from __future__ import annotations

import os

# Anchor to the project root (one level up from ck_engine/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WIN_DRIVE_ROOT = "V:\\"
LINUX_MOUNT_ROOT = "/mnt/ssd-storage"


def map_path(win_path: str) -> str:
    r"""Convert a Windows path (e.g. V:\Projects\Shot1) to the local Linux mount."""
    win_path = win_path.strip()
    if win_path.upper().startswith(WIN_DRIVE_ROOT.upper()):
        rel_path = win_path[len(WIN_DRIVE_ROOT) :]
        return os.path.join(LINUX_MOUNT_ROOT, rel_path).replace("\\", "/")
    return win_path
