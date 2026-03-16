"""Tests for backend.validators — frame count, mask normalization, write, and output dirs."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from ck_engine.config import Dir
from ck_engine.errors import FrameMismatchError, MaskChannelError, WriteFailureError
from ck_engine.validators import (
    ensure_output_dirs,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_counts,
    validate_write,
)


# ---------------------------------------------------------------------------
# TestValidateFrameCounts
# ---------------------------------------------------------------------------


class TestValidateFrameCounts:
    """Tests for validate_frame_counts."""

    def test_equal_counts_returns_count(self):
        assert validate_frame_counts("clip", 100, 100) == 100

    def test_input_gt_alpha_non_strict_returns_min(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ck_engine.validators"):
            result = validate_frame_counts("clip", 120, 100, strict=False)
        assert result == 100
        assert "mismatch" in caplog.text.lower()

    def test_alpha_gt_input_non_strict_returns_min(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ck_engine.validators"):
            result = validate_frame_counts("clip", 80, 100, strict=False)
        assert result == 80
        assert "mismatch" in caplog.text.lower()

    def test_mismatch_strict_raises_frame_mismatch_error(self):
        with pytest.raises(FrameMismatchError) as exc_info:
            validate_frame_counts("my_clip", 50, 60, strict=True)
        err = exc_info.value
        assert err.clip_name == "my_clip"
        assert err.input_count == 50
        assert err.alpha_count == 60

    def test_both_zero_returns_zero(self):
        assert validate_frame_counts("clip", 0, 0) == 0


# ---------------------------------------------------------------------------
# TestNormalizeMaskDtype
# ---------------------------------------------------------------------------


class TestNormalizeMaskDtype:
    """Tests for normalize_mask_dtype."""

    def test_uint8_to_float32(self):
        mask = np.full((4, 4), 255, dtype=np.uint8)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0)

    def test_uint16_to_float32(self):
        mask = np.full((4, 4), 65535, dtype=np.uint16)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0)

    def test_float64_to_float32(self):
        mask = np.array([[0.25, 0.75]], dtype=np.float64)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [[0.25, 0.75]])

    def test_float32_returned_as_is(self):
        mask = np.array([[0.5]], dtype=np.float32)
        result = normalize_mask_dtype(mask)
        assert result is mask  # identity, not a copy


# ---------------------------------------------------------------------------
# TestNormalizeMaskChannels
# ---------------------------------------------------------------------------


class TestNormalizeMaskChannels:
    """Tests for normalize_mask_channels."""

    def test_2d_passthrough(self):
        mask = np.ones((4, 4), dtype=np.float32)
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        assert result.shape == (4, 4)

    def test_3d_single_channel_squeezed(self):
        mask = np.ones((4, 4, 1), dtype=np.float32)
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        assert result.shape == (4, 4)

    def test_3d_three_channels_extracts_first(self):
        mask = np.zeros((4, 4, 3), dtype=np.float32)
        mask[:, :, 0] = 0.8
        mask[:, :, 1] = 0.2
        mask[:, :, 2] = 0.5
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        np.testing.assert_allclose(result, 0.8)

    def test_3d_four_channels_extracts_first(self):
        mask = np.zeros((4, 4, 4), dtype=np.float32)
        mask[:, :, 0] = 0.9
        result = normalize_mask_channels(mask)
        assert result.ndim == 2
        np.testing.assert_allclose(result, 0.9)

    def test_3d_zero_channels_raises_mask_channel_error(self):
        mask = np.empty((4, 4, 0), dtype=np.float32)
        with pytest.raises(MaskChannelError) as exc_info:
            normalize_mask_channels(mask, clip_name="bad_clip", frame_index=7)
        err = exc_info.value
        assert err.clip_name == "bad_clip"
        assert err.frame_index == 7
        assert err.channels == 0


# ---------------------------------------------------------------------------
# TestValidateWrite
# ---------------------------------------------------------------------------


class TestValidateWrite:
    """Tests for validate_write."""

    def test_true_no_exception(self):
        validate_write(True, "clip", 0, "/tmp/frame.png")

    def test_false_raises_write_failure_error(self):
        with pytest.raises(WriteFailureError) as exc_info:
            validate_write(False, "my_clip", 42, "/out/frame_042.exr")
        err = exc_info.value
        assert err.clip_name == "my_clip"
        assert err.frame_index == 42
        assert err.path == "/out/frame_042.exr"


# ---------------------------------------------------------------------------
# TestEnsureOutputDirs
# ---------------------------------------------------------------------------


class TestEnsureOutputDirs:
    """Tests for ensure_output_dirs."""

    def test_creates_all_expected_directories(self, tmp_path):
        dirs = ensure_output_dirs(str(tmp_path))
        for d in dirs.values():
            assert (tmp_path / d.removeprefix(str(tmp_path) + "/")).is_dir(), (
                f"Directory not created: {d}"
            )

    def test_returns_dict_with_correct_keys(self, tmp_path):
        dirs = ensure_output_dirs(str(tmp_path))
        assert set(dirs.keys()) == {"root", "fg", "matte", "comp", "processed"}
        assert dirs["root"].endswith(Dir.OUTPUT)
        assert dirs["fg"].endswith(Dir.FG)
        assert dirs["matte"].endswith(Dir.MATTE)
        assert dirs["comp"].endswith(Dir.COMP)
        assert dirs["processed"].endswith(Dir.PROCESSED)

    def test_idempotent(self, tmp_path):
        dirs1 = ensure_output_dirs(str(tmp_path))
        dirs2 = ensure_output_dirs(str(tmp_path))
        assert dirs1 == dirs2


# ---------------------------------------------------------------------------
# Ordering invariant: dtype THEN channels
# ---------------------------------------------------------------------------


class TestNormalizationOrdering:
    """Guard the load-bearing call order used in frame_io.py.

    normalize_mask_dtype must be called BEFORE normalize_mask_channels.

    With a uint8(255) 3-channel mask:
      - dtype first  -> divides by 255 -> 1.0, then channel extraction -> 1.0
      - channels first -> extracts raw uint8 channel (255), casts to float32 -> 255.0

    Reversing the order silently produces masks with values in [0, 255] instead
    of [0.0, 1.0], which corrupts all downstream compositing.
    """

    def test_dtype_then_channels_gives_normalized_values(self):
        mask = np.full((4, 4, 3), 255, dtype=np.uint8)
        step1 = normalize_mask_dtype(mask)           # uint8 -> float32 / 255
        step2 = normalize_mask_channels(step1)       # extract channel 0
        np.testing.assert_allclose(step2, 1.0)

    def test_channels_then_dtype_also_correct_but_documents_order(self):
        """Channels-first happens to work here because normalize_mask_channels
        casts to float32 (without scaling), then normalize_mask_dtype sees
        float32 and returns as-is. The result is 255.0 — NOT normalized.

        This test documents why the call order matters.
        """
        mask = np.full((4, 4, 3), 255, dtype=np.uint8)
        step1 = normalize_mask_channels(mask)        # extract ch0, cast to float32
        step2 = normalize_mask_dtype(step1)           # float32 -> returned as-is
        # Wrong! Values are 255.0 instead of 1.0.
        np.testing.assert_allclose(step2, 255.0)
