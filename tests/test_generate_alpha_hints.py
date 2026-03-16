"""Tests for backend.pipeline.generate_alpha_hints."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ck_engine.clip_state import ClipAsset, ClipEntry
from ck_engine.pipeline import AlphaMode, generate_alpha_hints


def _make_clip(tmp_path, name: str, *, frame_count: int = 10) -> ClipEntry:
    """Create a ClipEntry with a real directory and a stubbed input asset."""
    clip_dir = tmp_path / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    input_dir = clip_dir / "Input"
    input_dir.mkdir(exist_ok=True)

    asset = MagicMock(spec=ClipAsset)
    asset.path = str(input_dir)
    asset.frame_count = frame_count

    return ClipEntry(name=name, root_path=str(clip_dir), input_asset=asset)


@pytest.fixture()
def mock_generator():
    gen = MagicMock()
    gen.requires_mask = False
    gen.generate.return_value = 5
    return gen


@pytest.fixture()
def _patch_device():
    with patch("ck_engine.pipeline.resolve_device", return_value="cpu"):
        yield


# ---------------------------------------------------------------------------
# AlphaMode.SKIP
# ---------------------------------------------------------------------------


class TestAlphaModeSkip:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_skip_returns_zero(self, _rd, tmp_path):
        clip = _make_clip(tmp_path, "clip1")
        with patch("ck_engine.generators.get_generator") as mock_get:
            result = generate_alpha_hints([clip], model="gvm", alpha_mode=AlphaMode.SKIP)
        assert result == 0
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Model init failures
# ---------------------------------------------------------------------------


class TestModelInitFailure:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_import_error_propagates(self, _rd, tmp_path):
        clip = _make_clip(tmp_path, "clip1")
        with patch("ck_engine.generators.get_generator", side_effect=ImportError("no module")):
            with pytest.raises(ImportError, match="no module"):
                generate_alpha_hints([clip], model="gvm")

    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_runtime_error_propagates(self, _rd, tmp_path):
        clip = _make_clip(tmp_path, "clip1")
        with patch("ck_engine.generators.get_generator", side_effect=RuntimeError("bad init")):
            with pytest.raises(RuntimeError, match="bad init"):
                generate_alpha_hints([clip], model="gvm")


# ---------------------------------------------------------------------------
# Per-clip success / failure
# ---------------------------------------------------------------------------


class TestPerClipSuccessFailure:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_two_clips_both_succeed(self, _rd, tmp_path, mock_generator):
        clips = [_make_clip(tmp_path, "a"), _make_clip(tmp_path, "b")]
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            result = generate_alpha_hints(clips, model="gvm")
        assert result == 2
        assert mock_generator.generate.call_count == 2

    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_second_clip_fails(self, _rd, tmp_path, mock_generator):
        clips = [_make_clip(tmp_path, "a"), _make_clip(tmp_path, "b")]
        mock_generator.generate.side_effect = [5, Exception("boom")]
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            result = generate_alpha_hints(clips, model="gvm")
        assert result == 1


# ---------------------------------------------------------------------------
# Mask discovery
# ---------------------------------------------------------------------------


class TestMaskDiscovery:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_requires_mask_no_mask_dir(self, _rd, tmp_path):
        clip = _make_clip(tmp_path, "clip1")
        gen = MagicMock()
        gen.requires_mask = True
        with patch("ck_engine.generators.get_generator", return_value=gen):
            result = generate_alpha_hints([clip], model="videomama")
        assert result == 0
        gen.generate.assert_not_called()


# ---------------------------------------------------------------------------
# AlphaMode.REPLACE / FILL
# ---------------------------------------------------------------------------


class TestAlphaModes:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_replace_removes_existing_alpha_dir(self, _rd, tmp_path, mock_generator):
        clip = _make_clip(tmp_path, "clip1")
        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)
        # Put a sentinel file so we can verify deletion
        sentinel = os.path.join(alpha_dir, "old.png")
        with open(sentinel, "w") as f:
            f.write("x")

        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            result = generate_alpha_hints(
                [clip], model="gvm", alpha_mode=AlphaMode.REPLACE
            )

        assert result == 1
        # The old sentinel file should have been removed (rmtree then generator writes)
        assert not os.path.exists(sentinel)
        # Verify skip_existing=False was passed
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["skip_existing"] is False

    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_fill_passes_skip_existing(self, _rd, tmp_path, mock_generator):
        clip = _make_clip(tmp_path, "clip1")
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            result = generate_alpha_hints(
                [clip], model="gvm", alpha_mode=AlphaMode.FILL
            )
        assert result == 1
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["skip_existing"] is True


# ---------------------------------------------------------------------------
# Frame indices
# ---------------------------------------------------------------------------


class TestFrameIndices:
    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_start_and_end(self, _rd, tmp_path, mock_generator):
        clip = _make_clip(tmp_path, "clip1", frame_count=20)
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            generate_alpha_hints([clip], model="gvm", start=2, end=5)
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["frame_indices"] == range(2, 6)

    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_no_start_no_end(self, _rd, tmp_path, mock_generator):
        clip = _make_clip(tmp_path, "clip1", frame_count=20)
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            generate_alpha_hints([clip], model="gvm")
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["frame_indices"] is None

    @patch("ck_engine.pipeline.resolve_device", return_value="cpu")
    def test_start_only(self, _rd, tmp_path, mock_generator):
        clip = _make_clip(tmp_path, "clip1", frame_count=20)
        with patch("ck_engine.generators.get_generator", return_value=mock_generator):
            generate_alpha_hints([clip], model="gvm", start=1, end=None)
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["frame_indices"] == range(1, 20)
