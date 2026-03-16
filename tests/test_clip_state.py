"""Tests for ClipEntry state machine, asset resolution, and output queries."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ck_engine.clip_state import ClipAsset, ClipEntry, ClipState, _TRANSITIONS
from ck_engine.config import Dir
from ck_engine.errors import ClipScanError, InvalidStateTransitionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch_png(path: os.PathLike | str) -> None:
    """Create a tiny file with a .png extension (content doesn't matter)."""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")


def _make_clip(tmp_path, name: str = "shot", *, n_input: int = 0,
               n_alpha: int = 0, n_mask: int = 0) -> ClipEntry:
    """Build a clip directory with the given frame counts and return a ClipEntry."""
    root = tmp_path / name
    input_dir = root / Dir.INPUT
    alpha_dir = root / Dir.ALPHA_HINT
    mask_dir = root / Dir.VIDEOMAMA_HINT

    for d in (input_dir, alpha_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(n_input):
        _touch_png(input_dir / f"frame_{i:04d}.png")
    for i in range(n_alpha):
        _touch_png(alpha_dir / f"frame_{i:04d}.png")
    for i in range(n_mask):
        _touch_png(mask_dir / f"frame_{i:04d}.png")

    return ClipEntry(name=name, root_path=str(root))


# ===================================================================
# TestStateTransitions
# ===================================================================

class TestStateTransitions:
    """Validate the transition table — both valid and invalid moves."""

    # Enumerate every valid edge from the _TRANSITIONS dict.
    _valid_edges = [
        (src, dst)
        for src, targets in _TRANSITIONS.items()
        for dst in targets
    ]

    @pytest.mark.parametrize("src,dst", _valid_edges,
                             ids=[f"{s.value}->{d.value}" for s, d in _valid_edges])
    def test_valid_transition_succeeds(self, src: ClipState, dst: ClipState):
        clip = ClipEntry(name="c", root_path="/tmp/fake", state=src,
                         error_message="old err")
        clip.transition_to(dst)

        assert clip.state is dst
        if dst is not ClipState.ERROR:
            assert clip.error_message is None, (
                "error_message should be cleared on non-ERROR transitions"
            )

    def test_invalid_transition_raises(self):
        clip = ClipEntry(name="myshot", root_path="/tmp/fake", state=ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            clip.transition_to(ClipState.COMPLETE)

        err = exc_info.value
        assert err.clip_name == "myshot"
        assert err.current_state == "RAW"
        assert err.target_state == "COMPLETE"

    def test_multi_hop_chain(self):
        """RAW -> ERROR -> RAW -> READY is reachable across multiple transitions."""
        clip = ClipEntry(name="retry", root_path="/tmp/fake")
        assert clip.state is ClipState.RAW

        clip.transition_to(ClipState.ERROR)
        assert clip.state is ClipState.ERROR

        clip.transition_to(ClipState.RAW)
        assert clip.state is ClipState.RAW

        clip.transition_to(ClipState.READY)
        assert clip.state is ClipState.READY


# ===================================================================
# TestSetError
# ===================================================================

class TestSetError:
    """Verify set_error() transitions to ERROR and stores the message."""

    @pytest.mark.parametrize("src", [ClipState.RAW, ClipState.MASKED, ClipState.READY])
    def test_set_error_from_valid_source(self, src: ClipState):
        clip = ClipEntry(name="e", root_path="/tmp/fake", state=src)
        clip.set_error("something broke")

        assert clip.state is ClipState.ERROR
        assert clip.error_message == "something broke"

    def test_set_error_from_complete_raises(self):
        """COMPLETE -> ERROR is not in the transition table."""
        clip = ClipEntry(name="e", root_path="/tmp/fake", state=ClipState.COMPLETE)
        with pytest.raises(InvalidStateTransitionError):
            clip.set_error("should fail")

    def test_set_error_from_error_raises(self):
        """ERROR -> ERROR is not a valid self-transition."""
        clip = ClipEntry(name="e", root_path="/tmp/fake", state=ClipState.ERROR)
        with pytest.raises(InvalidStateTransitionError):
            clip.set_error("double fault")


# ===================================================================
# TestResolveState
# ===================================================================

class TestResolveState:
    """Verify _resolve_state() picks the correct state based on assets on disk."""

    def test_raw_no_assets(self, tmp_path):
        clip = _make_clip(tmp_path, n_input=3)
        clip.find_assets()
        assert clip.state is ClipState.RAW

    def test_masked_with_mask_hint(self, tmp_path):
        clip = _make_clip(tmp_path, n_input=3, n_mask=1)
        clip.find_assets()
        assert clip.state is ClipState.MASKED

    def test_ready_with_full_alpha(self, tmp_path):
        clip = _make_clip(tmp_path, n_input=3, n_alpha=3)
        clip.find_assets()
        assert clip.state is ClipState.READY

    def test_partial_alpha_stays_lower(self, tmp_path):
        """Alpha with fewer frames than input should NOT promote to READY."""
        clip = _make_clip(tmp_path, n_input=5, n_alpha=2)
        clip.find_assets()
        # Partial alpha -- should fall through to RAW (no mask)
        assert clip.state is ClipState.RAW

    def test_partial_alpha_with_mask_goes_masked(self, tmp_path):
        """Partial alpha + mask present -> MASKED (mask takes precedence over RAW)."""
        clip = _make_clip(tmp_path, n_input=5, n_alpha=2, n_mask=1)
        clip.find_assets()
        assert clip.state is ClipState.MASKED

    def test_no_input_raises(self, tmp_path):
        """Missing input media should raise ClipScanError."""
        root = tmp_path / "empty"
        (root / Dir.INPUT).mkdir(parents=True)
        clip = ClipEntry(name="empty", root_path=str(root))
        with pytest.raises(ClipScanError):
            clip.find_assets()


# ===================================================================
# TestCompletedStems
# ===================================================================

class TestCompletedStems:
    """Verify completed_stems() intersection logic."""

    def test_no_output_dir_returns_empty(self, tmp_path):
        clip = ClipEntry(name="x", root_path=str(tmp_path / "nonexistent"))
        assert clip.completed_stems() == set()

    def test_fg_and_matte_matching(self, tmp_path):
        """FG and Matte with identical stems -> full intersection."""
        root = tmp_path / "clip"
        fg_dir = root / Dir.OUTPUT / Dir.FG
        matte_dir = root / Dir.OUTPUT / Dir.MATTE
        fg_dir.mkdir(parents=True)
        matte_dir.mkdir(parents=True)

        for i in range(5):
            _touch_png(fg_dir / f"frame_{i:04d}.png")
            _touch_png(matte_dir / f"frame_{i:04d}.png")

        clip = ClipEntry(name="clip", root_path=str(root))
        stems = clip.completed_stems()
        expected = {f"frame_{i:04d}" for i in range(5)}
        assert stems == expected

    def test_fg_and_matte_partial_overlap(self, tmp_path):
        """FG has frames 0-4, Matte has 0-2 -> intersection is stems 0-2."""
        root = tmp_path / "clip"
        fg_dir = root / Dir.OUTPUT / Dir.FG
        matte_dir = root / Dir.OUTPUT / Dir.MATTE
        fg_dir.mkdir(parents=True)
        matte_dir.mkdir(parents=True)

        for i in range(5):
            _touch_png(fg_dir / f"frame_{i:04d}.png")
        for i in range(3):
            _touch_png(matte_dir / f"frame_{i:04d}.png")

        clip = ClipEntry(name="clip", root_path=str(root))
        stems = clip.completed_stems()
        expected = {f"frame_{i:04d}" for i in range(3)}
        assert stems == expected

    def test_non_image_files_ignored(self, tmp_path):
        """Files that are not images should not count as completed stems."""
        root = tmp_path / "clip"
        fg_dir = root / Dir.OUTPUT / Dir.FG
        matte_dir = root / Dir.OUTPUT / Dir.MATTE
        fg_dir.mkdir(parents=True)
        matte_dir.mkdir(parents=True)

        _touch_png(fg_dir / "frame_0000.png")
        (fg_dir / "frame_0001.txt").write_text("not an image")
        _touch_png(matte_dir / "frame_0000.png")

        clip = ClipEntry(name="clip", root_path=str(root))
        assert clip.completed_stems() == {"frame_0000"}


# ===================================================================
# TestHasOutputs
# ===================================================================

class TestHasOutputs:
    """Verify has_outputs property checks for real output content."""

    def test_no_output_dir(self, tmp_path):
        clip = ClipEntry(name="x", root_path=str(tmp_path / "nope"))
        assert clip.has_outputs is False

    def test_output_dir_all_empty_subdirs(self, tmp_path):
        root = tmp_path / "clip"
        for subdir in (Dir.FG, Dir.MATTE, Dir.COMP, Dir.PROCESSED):
            (root / Dir.OUTPUT / subdir).mkdir(parents=True)

        clip = ClipEntry(name="clip", root_path=str(root))
        assert clip.has_outputs is False

    def test_output_dir_with_content_in_fg(self, tmp_path):
        root = tmp_path / "clip"
        fg_dir = root / Dir.OUTPUT / Dir.FG
        fg_dir.mkdir(parents=True)
        _touch_png(fg_dir / "frame_0000.png")

        clip = ClipEntry(name="clip", root_path=str(root))
        assert clip.has_outputs is True

    def test_output_dir_with_content_in_matte_only(self, tmp_path):
        root = tmp_path / "clip"
        matte_dir = root / Dir.OUTPUT / Dir.MATTE
        matte_dir.mkdir(parents=True)
        _touch_png(matte_dir / "frame_0000.png")
        # Create other subdirs empty
        for subdir in (Dir.FG, Dir.COMP, Dir.PROCESSED):
            (root / Dir.OUTPUT / subdir).mkdir(parents=True, exist_ok=True)

        clip = ClipEntry(name="clip", root_path=str(root))
        assert clip.has_outputs is True
