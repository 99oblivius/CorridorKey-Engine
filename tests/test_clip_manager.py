"""Tests for clip_manager.py utility functions and ClipEntry discovery.

These tests verify the non-interactive parts of clip_manager: file type
detection, Windows→Linux path mapping, and the ClipEntry asset discovery
that scans directory trees to find Input/AlphaHint pairs.

No GPU, model weights, or interactive input required.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from ck_engine.clip_state import ClipAsset, ClipEntry
from ck_engine.config import Dir
from ck_engine.path_utils import map_path
from ck_engine.project import is_image_file, is_video_file

# ---------------------------------------------------------------------------
# is_image_file / is_video_file
# ---------------------------------------------------------------------------


class TestFileTypeDetection:
    """Verify extension-based file type helpers.

    These are used everywhere in clip_manager to decide how to read inputs.
    A missed extension means a valid frame silently disappears from the batch.
    """

    @pytest.mark.parametrize(
        "filename",
        [
            "frame.png",
            "SHOT_001.EXR",
            "plate.jpg",
            "ref.JPEG",
            "scan.tif",
            "deep.tiff",
            "comp.bmp",
        ],
    )
    def test_image_extensions_recognized(self, filename):
        assert is_image_file(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "frame.mp4",
            "CLIP.MOV",
            "take.avi",
            "rushes.mkv",
        ],
    )
    def test_video_extensions_recognized(self, filename):
        assert is_video_file(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "readme.txt",
            "notes.pdf",
            "project.nk",
            "scene.blend",
            ".DS_Store",
        ],
    )
    def test_non_media_rejected(self, filename):
        assert not is_image_file(filename)
        assert not is_video_file(filename)

    def test_image_is_not_video(self):
        """Image and video extensions must not overlap."""
        assert not is_video_file("frame.png")
        assert not is_video_file("plate.exr")

    def test_video_is_not_image(self):
        assert not is_image_file("clip.mp4")
        assert not is_image_file("rushes.mov")


# ---------------------------------------------------------------------------
# map_path
# ---------------------------------------------------------------------------


class TestMapPath:
    r"""Windows→Linux path mapping.

    The tool is designed for studios running a Linux render farm with
    Windows workstations.  V:\ maps to /mnt/ssd-storage.
    """

    def test_basic_mapping(self):
        result = map_path(r"V:\Projects\Shot1")
        assert result == "/mnt/ssd-storage/Projects/Shot1"

    def test_case_insensitive_drive_letter(self):
        result = map_path(r"v:\projects\shot1")
        assert result == "/mnt/ssd-storage/projects/shot1"

    def test_trailing_whitespace_stripped(self):
        result = map_path(r"  V:\Projects\Shot1  ")
        assert result == "/mnt/ssd-storage/Projects/Shot1"

    def test_backslashes_converted(self):
        result = map_path(r"V:\Deep\Nested\Path\Here")
        assert "\\" not in result

    def test_non_v_drive_passthrough(self):
        """Paths not on V: are returned as-is (may already be Linux paths)."""
        linux_path = "/mnt/other/data"
        assert map_path(linux_path) == linux_path

    def test_drive_root_only(self):
        result = map_path("V:\\")
        assert result == "/mnt/ssd-storage/"


# ---------------------------------------------------------------------------
# ClipAsset
# ---------------------------------------------------------------------------


class TestClipAsset:
    """ClipAsset wraps a directory of images or a video file and counts frames."""

    def test_sequence_frame_count(self, tmp_path):
        """Image sequence: frame count = number of image files in directory."""
        seq_dir = tmp_path / Dir.INPUT
        seq_dir.mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(str(seq_dir / f"frame_{i:04d}.png"), tiny)

        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 5

    def test_sequence_ignores_non_image_files(self, tmp_path):
        """Non-image files (thumbs.db, .nk, etc.) should not be counted."""
        seq_dir = tmp_path / Dir.INPUT
        seq_dir.mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "frame_0000.png"), tiny)
        (seq_dir / "thumbs.db").write_text("junk")
        (seq_dir / "notes.txt").write_text("notes")

        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 1

    def test_empty_sequence(self, tmp_path):
        """Empty directory → 0 frames."""
        seq_dir = tmp_path / Dir.INPUT
        seq_dir.mkdir()
        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 0


# ---------------------------------------------------------------------------
# ClipEntry.find_assets
# ---------------------------------------------------------------------------


class TestClipEntryFindAssets:
    """ClipEntry.find_assets() discovers Input and AlphaHint from a shot directory.

    This is the core discovery logic that decides what's ready for inference
    vs. what still needs alpha generation.
    """

    def test_finds_image_sequence_input(self, tmp_clip_dir):
        """shot_a has Input/ with 2 PNGs → input_asset is a sequence."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        assert entry.input_asset is not None
        assert entry.input_asset.asset_type == "sequence"
        assert entry.input_asset.frame_count == 2

    def test_finds_alpha_hint(self, tmp_clip_dir):
        """shot_a has AlphaHint/ with 2 PNGs → alpha_asset is populated."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        assert entry.alpha_asset is not None
        assert entry.alpha_asset.asset_type == "sequence"
        assert entry.alpha_asset.frame_count == 2

    def test_empty_alpha_hint_is_none(self, tmp_clip_dir):
        """shot_b has empty AlphaHint/ → alpha_asset is None (needs generation)."""
        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()
        assert entry.input_asset is not None
        assert entry.alpha_asset is None

    def test_missing_input_raises(self, tmp_path):
        """A shot with no Input directory raises ClipScanError."""
        from ck_engine.errors import ClipScanError

        empty_shot = tmp_path / "empty_shot"
        empty_shot.mkdir()
        entry = ClipEntry(name="empty_shot", root_path=str(empty_shot))
        with pytest.raises(ClipScanError, match="no usable media in Input"):
            entry.find_assets()

    def test_empty_input_dir_raises(self, tmp_path):
        """An empty Input/ directory raises ClipScanError."""
        from ck_engine.errors import ClipScanError

        shot = tmp_path / "bad_shot"
        (shot / Dir.INPUT).mkdir(parents=True)
        entry = ClipEntry(name="bad_shot", root_path=str(shot))
        with pytest.raises(ClipScanError, match="no usable media in Input"):
            entry.find_assets()


