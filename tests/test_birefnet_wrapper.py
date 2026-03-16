"""Tests for BiRefNet alpha generator wrapper logic.

Tests cover frame counting, frame iteration, and the generate() pipeline
without loading any model weights -- BiRefNetProcessor is fully mocked.
"""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import cv2
import numpy as np
import pytest

from ck_engine.generators.birefnet.wrapper import BiRefNetAlphaGenerator, BiRefNetProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pngs(directory: Path, count: int, *, size: int = 4) -> list[Path]:
    """Create *count* tiny PNG files in *directory* and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        p = directory / f"frame_{i:04d}.png"
        img = np.full((size, size, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _make_generator(mock_processor: unittest.mock.MagicMock | None = None) -> BiRefNetAlphaGenerator:
    """Construct a BiRefNetAlphaGenerator without loading model weights."""
    with unittest.mock.patch.object(BiRefNetProcessor, "__init__", lambda self, **kw: None):
        gen = BiRefNetAlphaGenerator(device="cpu")
    if mock_processor is not None:
        gen._processor = mock_processor
    return gen


# ===========================================================================
# TestCountFrames
# ===========================================================================

class TestCountFrames:
    """Tests for BiRefNetAlphaGenerator._count_frames."""

    def test_counts_all_pngs(self, tmp_path: Path) -> None:
        _make_pngs(tmp_path, 5)
        gen = _make_generator()
        assert gen._count_frames(tmp_path) == 5

    def test_counts_with_indices(self, tmp_path: Path) -> None:
        _make_pngs(tmp_path, 5)
        gen = _make_generator()
        assert gen._count_frames(tmp_path, indices={0, 2, 4}) == 3

    def test_empty_dir(self, tmp_path: Path) -> None:
        gen = _make_generator()
        assert gen._count_frames(tmp_path) == 0


# ===========================================================================
# TestIterFrames
# ===========================================================================

class TestIterFrames:
    """Tests for BiRefNetAlphaGenerator._iter_frames."""

    def test_yields_all_frames(self, tmp_path: Path) -> None:
        _make_pngs(tmp_path, 3)
        gen = _make_generator()
        results = list(gen._iter_frames(tmp_path))
        assert len(results) == 3
        for stem, rgb in results:
            assert isinstance(stem, str)
            assert isinstance(rgb, np.ndarray)
            assert rgb.ndim == 3 and rgb.shape[2] == 3

    def test_filters_by_indices(self, tmp_path: Path) -> None:
        _make_pngs(tmp_path, 3)
        gen = _make_generator()
        results = list(gen._iter_frames(tmp_path, indices={0, 2}))
        assert len(results) == 2
        stems = [s for s, _ in results]
        assert "frame_0000" in stems
        assert "frame_0002" in stems

    def test_out_of_range_index_yields_nothing(self, tmp_path: Path) -> None:
        _make_pngs(tmp_path, 3)
        gen = _make_generator()
        results = list(gen._iter_frames(tmp_path, indices={5}))
        assert len(results) == 0


# ===========================================================================
# TestGenerate
# ===========================================================================

class TestGenerate:
    """Tests for BiRefNetAlphaGenerator.generate."""

    @pytest.fixture()
    def input_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "input"
        _make_pngs(d, 3)
        return d

    @pytest.fixture()
    def output_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "output"
        d.mkdir()
        return d

    @pytest.fixture()
    def mock_proc(self) -> unittest.mock.MagicMock:
        proc = unittest.mock.MagicMock()
        proc.process_frame.return_value = np.zeros((4, 4), dtype=np.uint8)
        return proc

    # -- normal run ---------------------------------------------------------

    def test_normal_run(
        self,
        input_dir: Path,
        output_dir: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        gen = _make_generator(mock_proc)
        written = gen.generate(str(input_dir), str(output_dir))
        assert written == 3
        assert mock_proc.process_frame.call_count == 3
        output_files = sorted(output_dir.glob("*.png"))
        assert len(output_files) == 3

    # -- skip_existing ------------------------------------------------------

    def test_skip_existing_true(
        self,
        input_dir: Path,
        output_dir: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        # Pre-create one output file so it gets skipped.
        pre = output_dir / "frame_0001.png"
        cv2.imwrite(str(pre), np.zeros((4, 4), dtype=np.uint8))

        gen = _make_generator(mock_proc)
        written = gen.generate(str(input_dir), str(output_dir), skip_existing=True)
        assert written == 2
        # Processor must NOT have been called for frame_0001.
        called_args = [
            call.args[0] for call in mock_proc.process_frame.call_args_list
        ]
        assert mock_proc.process_frame.call_count == 2

    def test_skip_existing_false_overwrites(
        self,
        input_dir: Path,
        output_dir: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        pre = output_dir / "frame_0001.png"
        cv2.imwrite(str(pre), np.zeros((4, 4), dtype=np.uint8))

        gen = _make_generator(mock_proc)
        written = gen.generate(str(input_dir), str(output_dir), skip_existing=False)
        assert written == 3
        assert mock_proc.process_frame.call_count == 3

    # -- frame_indices ------------------------------------------------------

    def test_frame_indices(
        self,
        input_dir: Path,
        output_dir: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        gen = _make_generator(mock_proc)
        written = gen.generate(str(input_dir), str(output_dir), frame_indices=[1])
        assert written == 1
        assert mock_proc.process_frame.call_count == 1

    # -- empty input --------------------------------------------------------

    def test_empty_input(
        self,
        tmp_path: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        empty = tmp_path / "empty_in"
        empty.mkdir()
        out = tmp_path / "empty_out"

        gen = _make_generator(mock_proc)
        written = gen.generate(str(empty), str(out))
        assert written == 0
        mock_proc.process_frame.assert_not_called()

    # -- on_progress callback -----------------------------------------------

    def test_on_progress_callback(
        self,
        input_dir: Path,
        output_dir: Path,
        mock_proc: unittest.mock.MagicMock,
    ) -> None:
        progress_calls: list[tuple[int, int]] = []

        def _cb(current: int, total: int) -> None:
            progress_calls.append((current, total))

        gen = _make_generator(mock_proc)
        gen.generate(str(input_dir), str(output_dir), on_progress=_cb)

        assert len(progress_calls) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
