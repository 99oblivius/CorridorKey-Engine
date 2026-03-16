"""Tests for GVMAlphaGenerator.generate() — skip_existing, file renaming,
frame_indices deletion, and return value.

GVMProcessor requires model weights, so it is mocked entirely.
"""

from __future__ import annotations

import os
import unittest.mock

import cv2
import numpy as np
import pytest

from ck_engine.generators.gvm.wrapper import GVMAlphaGenerator, GVMProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path: str) -> None:
    """Write a tiny valid PNG to *path*."""
    cv2.imwrite(path, np.zeros((4, 4), dtype=np.uint8))


def _make_input_dir(tmp_path, names: list[str]) -> str:
    """Create an input directory with real PNG files named after *names*."""
    d = tmp_path / "input"
    d.mkdir(exist_ok=True)
    for name in names:
        _make_png(str(d / name))
    return str(d)


def _build_generator(mock_processor) -> GVMAlphaGenerator:
    """Construct a GVMAlphaGenerator without loading model weights."""
    with unittest.mock.patch.object(GVMProcessor, "__init__", lambda self, **kw: None):
        gen = GVMAlphaGenerator(device="cpu")
        gen._processor = mock_processor
    return gen


def _process_side_effect(output_names: list[str]):
    """Return a side_effect callable for ``process_sequence`` that creates
    stub PNG files with the given *output_names* inside *direct_output_dir*.
    """
    def _side_effect(*args, **kwargs):
        out_dir = kwargs.get("direct_output_dir")
        if out_dir is None:
            raise ValueError("direct_output_dir not passed")
        os.makedirs(out_dir, exist_ok=True)
        for name in output_names:
            _make_png(os.path.join(out_dir, name))
    return _side_effect


# ---------------------------------------------------------------------------
# TestSkipExisting
# ---------------------------------------------------------------------------

class TestSkipExisting:
    """Skip-existing logic: all-or-nothing for the temporal model."""

    def test_skip_when_output_has_enough_pngs(self, tmp_path):
        """Output dir already has >= expected count -> process_sequence NOT called."""
        input_dir = _make_input_dir(tmp_path, ["frame_001.png", "frame_002.png", "frame_003.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # Pre-populate output with 3 PNGs
        for name in ["frame_001.png", "frame_002.png", "frame_003.png"]:
            _make_png(str(out_dir / name))

        mock_proc = unittest.mock.MagicMock()
        gen = _build_generator(mock_proc)

        result = gen.generate(input_dir, str(out_dir), skip_existing=True)

        mock_proc.process_sequence.assert_not_called()
        assert result == 3

    def test_not_skipped_when_output_has_fewer_pngs(self, tmp_path):
        """Output dir has fewer PNGs than expected -> process_sequence IS called."""
        input_dir = _make_input_dir(tmp_path, ["frame_001.png", "frame_002.png", "frame_003.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # Only 1 existing PNG
        _make_png(str(out_dir / "frame_001.png"))

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        result = gen.generate(input_dir, str(out_dir), skip_existing=True)

        mock_proc.process_sequence.assert_called_once()
        assert result >= 3

    def test_skip_with_frame_indices_limiting_expected_count(self, tmp_path):
        """frame_indices reduces expected_count, so existing 2 PNGs suffice."""
        input_dir = _make_input_dir(tmp_path, ["a.png", "b.png", "c.png", "d.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # 2 existing PNGs
        _make_png(str(out_dir / "a.png"))
        _make_png(str(out_dir / "b.png"))

        mock_proc = unittest.mock.MagicMock()
        gen = _build_generator(mock_proc)

        # frame_indices=[0, 2] -> expected_count = 2
        result = gen.generate(
            input_dir, str(out_dir),
            skip_existing=True,
            frame_indices=[0, 2],
        )

        mock_proc.process_sequence.assert_not_called()
        assert result == 2


# ---------------------------------------------------------------------------
# TestFileRenaming
# ---------------------------------------------------------------------------

class TestFileRenaming:
    """GVM outputs numbered files (00000.png, ...) which are renamed to match
    the input stems."""

    def test_output_files_renamed_to_input_stems(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["frame_001.png", "frame_002.png", "frame_003.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        gen.generate(input_dir, str(out_dir))

        output_files = sorted(os.listdir(str(out_dir)))
        assert output_files == ["frame_001.png", "frame_002.png", "frame_003.png"]

    def test_more_gvm_files_than_input_stems(self, tmp_path):
        """Extra GVM output files beyond the number of input stems are left
        with their original names (break guard in the renaming loop)."""
        input_dir = _make_input_dir(tmp_path, ["frame_001.png", "frame_002.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        # GVM produced 4 files but input only has 2 stems
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png", "00003.png"]
        )
        gen = _build_generator(mock_proc)

        gen.generate(input_dir, str(out_dir))

        output_files = sorted(os.listdir(str(out_dir)))
        # First two renamed, last two keep original names
        assert "frame_001.png" in output_files
        assert "frame_002.png" in output_files
        assert "00002.png" in output_files
        assert "00003.png" in output_files
        assert len(output_files) == 4


# ---------------------------------------------------------------------------
# TestFrameIndicesDeletion
# ---------------------------------------------------------------------------

class TestFrameIndicesDeletion:
    """frame_indices filters output frames by deleting non-selected indices."""

    def test_frame_indices_deletes_non_selected(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["a.png", "b.png", "c.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        # Keep only indices 0 and 2 -> index 1 ("b.png") is deleted
        gen.generate(input_dir, str(out_dir), frame_indices=[0, 2])

        output_files = sorted(os.listdir(str(out_dir)))
        assert "a.png" in output_files
        assert "c.png" in output_files
        assert "b.png" not in output_files
        assert len(output_files) == 2

    def test_frame_indices_none_no_deletions(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["a.png", "b.png", "c.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        gen.generate(input_dir, str(out_dir), frame_indices=None)

        output_files = sorted(os.listdir(str(out_dir)))
        assert len(output_files) == 3


# ---------------------------------------------------------------------------
# TestReturnValue
# ---------------------------------------------------------------------------

class TestReturnValue:
    """generate() returns the count of .png files in output_dir after all
    operations (renaming + deletion)."""

    def test_returns_png_count_after_all_operations(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["x.png", "y.png", "z.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        result = gen.generate(input_dir, str(out_dir), frame_indices=[1])

        # Only index 1 kept
        assert result == 1

    def test_returns_full_count_without_frame_indices(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["a.png", "b.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png"]
        )
        gen = _build_generator(mock_proc)

        result = gen.generate(input_dir, str(out_dir))

        assert result == 2

    def test_on_progress_called_with_total(self, tmp_path):
        input_dir = _make_input_dir(tmp_path, ["a.png", "b.png", "c.png"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_proc = unittest.mock.MagicMock()
        mock_proc.process_sequence.side_effect = _process_side_effect(
            ["00000.png", "00001.png", "00002.png"]
        )
        gen = _build_generator(mock_proc)

        progress = unittest.mock.MagicMock()
        result = gen.generate(input_dir, str(out_dir), on_progress=progress)

        progress.assert_called_once_with(result, result)
