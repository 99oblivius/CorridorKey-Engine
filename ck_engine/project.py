"""Project folder management -- creation, scanning, and metadata.

A project is a timestamped container holding one or more clips:
    Projects/
        260301_093000_Woman_Jumps/
            clips/
                Woman_Jumps/                (ClipEntry.root_path -> here)
                    Input/                  (image sequence or video)
                    AlphaHint/
                    VideoMamaMaskHint/      (optional, user-provided)
                    Output/FG/ Matte/ Comp/ Processed/
                Man_Walks/
                    Input/...

Legacy v1 format (no clips/ dir) is still supported for backward compat.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from datetime import datetime

from .config import IMAGE_EXTS, VIDEO_EXTS, Dir
from .natural_sort import natsorted

logger = logging.getLogger(__name__)
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)"

def projects_root() -> str:
    """Return the Projects root directory, creating it if needed."""
    if getattr(sys, "frozen", False):
        root = os.path.join(os.path.dirname(sys.executable), "Projects")
    else:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Projects")
    os.makedirs(root, exist_ok=True)
    return root


def sanitize_stem(filename: str, max_len: int = 60) -> str:
    """Clean a filename stem for use in folder names."""
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:max_len]


def create_project(
    source_video_paths: str | list[str],
    *,
    copy_source: bool = True,
    display_name: str | None = None,
) -> str:
    """Create a new project folder for one or more source videos.

    Creates a v2 project with a ``clips/`` subdirectory.  Each video
    gets its own clip subfolder inside ``clips/``.

    Creates: Projects/YYMMDD_HHMMSS_{stem}/clips/{clip_stem}/Input/...
    """
    if isinstance(source_video_paths, str):
        source_video_paths = [source_video_paths]
    if not source_video_paths:
        raise ValueError("At least one source video path is required")

    root = projects_root()

    if display_name and display_name.strip():
        clean = display_name.strip()
        name_stem = re.sub(r"[^\w\-]", "_", clean)
        name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
    else:
        first_filename = os.path.basename(source_video_paths[0])
        name_stem = sanitize_stem(first_filename)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{name_stem}"

    project_dir = os.path.join(root, folder_name)
    if os.path.exists(project_dir):
        for i in range(2, 100):
            candidate = os.path.join(root, f"{folder_name}_{i}")
            if not os.path.exists(candidate):
                project_dir = candidate
                break
        else:
            raise RuntimeError(
                f"Could not find unique directory name for '{folder_name}' in '{root}'"
            )

    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    for video_path in source_video_paths:
        _create_clip_folder(clips_dir, video_path, copy_source=copy_source)

    return project_dir


def add_clips_to_project(
    project_dir: str,
    source_video_paths: list[str],
    *,
    copy_source: bool = True,
) -> list[str]:
    """Add new clips to an existing project."""
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    new_paths: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(clips_dir, video_path, copy_source=copy_source)
        new_paths.append(os.path.join(clips_dir, clip_name))

    return new_paths


def _create_clip_folder(
    clips_dir: str,
    video_path: str,
    *,
    copy_source: bool = True,
) -> str:
    """Create a single clip subfolder inside clips_dir.

    Returns the clip folder name (not full path).
    """
    filename = os.path.basename(video_path)
    clip_name = sanitize_stem(filename)

    clip_dir = os.path.join(clips_dir, clip_name)
    if os.path.exists(clip_dir):
        for i in range(2, 100):
            candidate = os.path.join(clips_dir, f"{clip_name}_{i}")
            if not os.path.exists(candidate):
                clip_dir = candidate
                clip_name = f"{clip_name}_{i}"
                break
        else:
            raise RuntimeError(
                f"Could not find unique directory name for '{clip_name}' in '{clips_dir}'"
            )

    input_dir = os.path.join(clip_dir, Dir.INPUT)
    os.makedirs(input_dir, exist_ok=True)

    if copy_source:
        target = os.path.join(input_dir, filename)
        if not os.path.isfile(target):
            shutil.copy2(video_path, target)
            logger.info(f"Copied source video: {video_path} -> {target}")
    else:
        logger.info(f"Referencing source video in place: {video_path}")

    os.makedirs(os.path.join(clip_dir, Dir.ALPHA_HINT), exist_ok=True)

    return clip_name


def get_clip_dirs(project_dir: str) -> list[str]:
    """Return absolute paths to all clip subdirectories in a project."""
    clips_dir = os.path.join(project_dir, "clips")
    if os.path.isdir(clips_dir):
        return natsorted(
            os.path.join(clips_dir, d)
            for d in os.listdir(clips_dir)
            if os.path.isdir(os.path.join(clips_dir, d)) and not d.startswith(".") and not d.startswith("_")
        )
    return [project_dir]


def is_v2_project(project_dir: str) -> bool:
    """Check if a project uses the v2 nested clips structure."""
    return os.path.isdir(os.path.join(project_dir, "clips"))


def is_video_file(filename: str) -> bool:
    """Check if a filename has a video extension."""
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTS


def is_image_file(filename: str) -> bool:
    """Check if a filename has an image extension."""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS
