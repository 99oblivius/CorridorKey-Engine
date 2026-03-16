"""AlphaGenerator protocol — common interface for all alpha hint generators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class AlphaGenerator(Protocol):
    """Common interface for all alpha hint generators.

    Each generator wraps a specific model (GVM, BiRefNet, VideoMaMa) and
    exposes a uniform ``generate()`` method that reads frames from disk
    and writes grayscale alpha-hint PNGs.
    """

    name: str          # "gvm", "birefnet", "videomama"
    is_temporal: bool   # True for GVM, VideoMaMa; False for BiRefNet
    requires_mask: bool # True for VideoMaMa only

    def generate(
        self,
        input_dir: str,
        output_dir: str,
        *,
        mask_dir: str | None = None,
        frame_indices: Sequence[int] | None = None,
        skip_existing: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Generate grayscale alpha-hint PNGs into output_dir.

        Parameters
        ----------
        input_dir : str
            Path to image sequence directory or video file.
        output_dir : str
            Directory to write output PNGs. Created if missing.
        mask_dir : str | None
            Path to mask sequence/video. Required when requires_mask is True.
        frame_indices : list[int] | None
            0-based indices of frames to write. None means all frames.
            Temporal models still process the full sequence internally.
        skip_existing : bool
            When True, skip frames whose output file already exists.
            Only effective for frame-independent models.
        on_progress : Callable[[int, int], None] | None
            Called with (frames_done, total_frames).

        Returns
        -------
        int
            Number of frames written.
        """
        ...
