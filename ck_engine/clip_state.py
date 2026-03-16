"""Clip entry data model and state machine.

State Machine:
    RAW        — Input asset found, no alpha hint yet
    MASKED     — User mask provided (for VideoMaMa workflow)
    READY      — Alpha hint available (from GVM or VideoMaMa), ready for inference
    COMPLETE   — Inference outputs written
    ERROR      — Processing failed (can retry)

Transitions:
    RAW → MASKED       (user provides VideoMaMa mask)
    RAW → READY        (GVM auto-generates alpha)
    RAW → ERROR        (GVM/scan fails)
    MASKED → READY     (VideoMaMa generates alpha from user mask)
    MASKED → ERROR     (VideoMaMa fails)
    READY → COMPLETE   (inference succeeds)
    READY → ERROR      (inference fails)
    ERROR → RAW        (retry from scratch)
    ERROR → MASKED     (retry with mask)
    ERROR → READY      (retry inference)
    COMPLETE → READY   (reprocess with different params)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum

from .config import Dir
from .errors import ClipScanError, InvalidStateTransitionError
from .natural_sort import natsorted
from .project import is_image_file as _is_image_file
from .project import is_video_file as _is_video_file

logger = logging.getLogger(__name__)


class ClipState(Enum):
    RAW = "RAW"
    MASKED = "MASKED"
    READY = "READY"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.RAW: {ClipState.MASKED, ClipState.READY, ClipState.ERROR},
    ClipState.MASKED: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},  # reprocess with different params
    ClipState.ERROR: {ClipState.RAW, ClipState.MASKED, ClipState.READY},
}


@dataclass
class ClipAsset:
    """Represents an input source — either an image sequence directory or a video file."""

    path: str
    asset_type: str  # 'sequence' or 'video'
    frame_count: int = 0

    def __post_init__(self) -> None:
        self._calculate_length()

    def _calculate_length(self) -> None:
        if self.asset_type == "sequence":
            if os.path.isdir(self.path):
                self.frame_count = sum(1 for f in os.listdir(self.path) if _is_image_file(f))
            else:
                self.frame_count = 0
        elif self.asset_type == "video":
            try:
                import cv2

                cap = cv2.VideoCapture(self.path)
                if cap.isOpened():
                    self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if self.frame_count == 0:
                        logger.warning(f"Video reports 0 frames, file may be corrupted: {self.path}")
                cap.release()
            except Exception as e:
                logger.debug(f"Video frame count detection failed for {self.path}: {e}")
                self.frame_count = 0

    def get_frame_files(self) -> list[str]:
        """Return naturally sorted list of frame filenames for sequence assets.

        Uses natural sort so frame_2 sorts before frame_10 (not lexicographic).
        """
        if self.asset_type != "sequence" or not os.path.isdir(self.path):
            return []
        return natsorted([f for f in os.listdir(self.path) if _is_image_file(f)])

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON transport."""
        return {
            "path": self.path,
            "asset_type": self.asset_type,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClipAsset:
        """Reconstruct from dict WITHOUT triggering filesystem I/O."""
        obj = object.__new__(cls)
        obj.path = d["path"]
        obj.asset_type = d["asset_type"]
        obj.frame_count = d.get("frame_count", 0)
        return obj


def _detect_dir_asset(dir_path: str) -> ClipAsset | None:
    """Detect whether a directory contains an image sequence or a single video.

    If the directory has image files, it's treated as a sequence.
    If it has only video file(s) and no images, the first video is used.
    Returns None if the directory is empty or has no recognizable media.
    """
    files = os.listdir(dir_path)
    if any(_is_image_file(f) for f in files):
        return ClipAsset(dir_path, "sequence")
    first_video = next((f for f in files if _is_video_file(f)), None)
    if first_video:
        return ClipAsset(os.path.join(dir_path, first_video), "video")
    return None


@dataclass
class ClipEntry:
    """A single shot/clip with its assets and processing state."""

    name: str
    root_path: str
    state: ClipState = ClipState.RAW
    input_asset: ClipAsset | None = None
    alpha_asset: ClipAsset | None = None
    mask_asset: ClipAsset | None = None  # User-provided VideoMaMa mask
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    _processing: bool = field(default=False, repr=False)  # lock: watcher must not reclassify

    @property
    def is_processing(self) -> bool:
        """True while a GPU job is actively working on this clip."""
        return self._processing

    def set_processing(self, value: bool) -> None:
        """Set processing lock. Watcher skips reclassification while True."""
        self._processing = value

    def transition_to(self, new_state: ClipState) -> None:
        """Attempt a state transition. Raises InvalidStateTransitionError if not allowed."""
        if new_state not in _TRANSITIONS.get(self.state, set()):
            raise InvalidStateTransitionError(self.name, self.state.value, new_state.value)
        old = self.state
        self.state = new_state
        if new_state != ClipState.ERROR:
            self.error_message = None
        logger.debug(f"Clip '{self.name}': {old.value} -> {new_state.value}")

    def set_error(self, message: str) -> None:
        """Transition to ERROR state with a message."""
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    @property
    def output_dir(self) -> str:
        return os.path.join(self.root_path, Dir.OUTPUT)

    @property
    def has_outputs(self) -> bool:
        """Check if output directory exists with content."""
        out = self.output_dir
        if not os.path.isdir(out):
            return False
        for subdir in (Dir.FG, Dir.MATTE, Dir.COMP, Dir.PROCESSED):
            d = os.path.join(out, subdir)
            if os.path.isdir(d) and os.listdir(d):
                return True
        return False

    def completed_frame_count(self) -> int:
        """Count existing output frames for resume support.

        Manifest-aware: reads .corridorkey_manifest.json to determine which
        outputs were enabled. Falls back to FG+Matte intersection if no manifest.
        """
        return len(self.completed_stems())

    def completed_stems(self) -> set[str]:
        """Return set of frame stems that have all enabled outputs complete.

        Reads the run manifest to determine which outputs to check.
        Falls back to FG+Matte intersection if no manifest exists.
        """
        manifest = self._read_manifest()
        if manifest:
            enabled = manifest.get("enabled_outputs", [])
        else:
            enabled = ["fg", "matte"]

        dir_map = {
            "fg": os.path.join(self.output_dir, Dir.FG),
            "matte": os.path.join(self.output_dir, Dir.MATTE),
            "comp": os.path.join(self.output_dir, Dir.COMP),
            "processed": os.path.join(self.output_dir, Dir.PROCESSED),
        }

        stem_sets = []
        for output_name in enabled:
            d = dir_map.get(output_name)
            if d and os.path.isdir(d):
                stems = {os.path.splitext(f)[0] for f in os.listdir(d) if _is_image_file(f)}
                stem_sets.append(stems)
            else:
                # Required dir missing -> no complete frames
                return set()

        if not stem_sets:
            return set()

        # Intersection: frame complete only if ALL enabled outputs exist
        result = stem_sets[0]
        for s in stem_sets[1:]:
            result &= s
        return result

    def _read_manifest(self) -> dict | None:
        """Read the run manifest if it exists."""
        manifest_path = os.path.join(self.output_dir, ".corridorkey_manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        try:
            import json

            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read manifest at {manifest_path}: {e}")
            return None

    def find_assets(self) -> None:
        """Scan the clip directory for Input, AlphaHint, and VideoMamaMaskHint assets.

        Each folder can contain either an image sequence or a video file.
        """
        input_dir = os.path.join(self.root_path, Dir.INPUT)
        alpha_dir = os.path.join(self.root_path, Dir.ALPHA_HINT)
        mask_dir = os.path.join(self.root_path, Dir.VIDEOMAMA_HINT)

        # Input asset
        if os.path.isdir(input_dir) and os.listdir(input_dir):
            self.input_asset = _detect_dir_asset(input_dir)
        if self.input_asset is None:
            raise ClipScanError(f"Clip '{self.name}': no usable media in Input/.")

        # Alpha hint asset
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            self.alpha_asset = _detect_dir_asset(alpha_dir)

        # VideoMaMa mask hint
        if os.path.isdir(mask_dir) and os.listdir(mask_dir):
            self.mask_asset = _detect_dir_asset(mask_dir)

        # Determine initial state
        self._resolve_state()

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON transport."""
        return {
            "name": self.name,
            "root_path": self.root_path,
            "state": self.state.value,
            "input_asset": self.input_asset.to_dict() if self.input_asset else None,
            "alpha_asset": self.alpha_asset.to_dict() if self.alpha_asset else None,
            "mask_asset": self.mask_asset.to_dict() if self.mask_asset else None,
            "warnings": list(self.warnings),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClipEntry:
        """Reconstruct from dict without filesystem access."""
        return cls(
            name=d["name"],
            root_path=d["root_path"],
            state=ClipState(d.get("state", "RAW")),
            input_asset=ClipAsset.from_dict(d["input_asset"]) if d.get("input_asset") else None,
            alpha_asset=ClipAsset.from_dict(d["alpha_asset"]) if d.get("alpha_asset") else None,
            mask_asset=ClipAsset.from_dict(d["mask_asset"]) if d.get("mask_asset") else None,
            warnings=d.get("warnings", []),
            error_message=d.get("error_message"),
        )

    def _resolve_state(self) -> None:
        """Set state based on what assets are present on disk.

        Priority (highest first):
          COMPLETE  — all input frames have matching outputs (manifest-aware)
          READY     — AlphaHint exists with matching frame count
          MASKED    — VideoMaMa mask hint exists
          RAW       — input exists, no alpha/mask/output
        """
        # Check COMPLETE first: outputs exist and cover all input frames
        if self.alpha_asset is not None and self.input_asset is not None:
            completed = self.completed_stems()
            if completed and len(completed) >= self.input_asset.frame_count:
                self.state = ClipState.COMPLETE
                return

        # READY: AlphaHint must cover ALL input frames (not partial)
        if self.alpha_asset is not None:
            if self.input_asset is not None and self.alpha_asset.frame_count < self.input_asset.frame_count:
                logger.info(
                    f"Clip '{self.name}': partial alpha "
                    f"({self.alpha_asset.frame_count}/{self.input_asset.frame_count}), "
                    f"staying at lower state"
                )
            else:
                self.state = ClipState.READY
                return

        if self.mask_asset is not None:
            self.state = ClipState.MASKED
        else:
            self.state = ClipState.RAW


def scan_clip(path: str) -> ClipEntry:
    """Scan a clip directory and return a fully-populated ClipEntry.

    This is the canonical way to create a ClipEntry from a filesystem path.
    The entry will have its assets discovered and state resolved.
    """
    name = os.path.basename(path)
    entry = ClipEntry(name=name, root_path=path)
    entry.find_assets()
    return entry
