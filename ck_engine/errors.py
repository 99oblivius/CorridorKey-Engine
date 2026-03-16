"""Typed exceptions for the CorridorKey backend."""


class CorridorKeyError(Exception):
    """Base exception for all CorridorKey backend errors."""


class ClipScanError(CorridorKeyError):
    """Raised when a clip directory cannot be scanned or is malformed."""


class FrameMismatchError(CorridorKeyError):
    """Raised when input and alpha frame counts don't match."""

    def __init__(self, clip_name: str, input_count: int, alpha_count: int) -> None:
        self.clip_name = clip_name
        self.input_count = input_count
        self.alpha_count = alpha_count
        super().__init__(f"Clip '{clip_name}': frame count mismatch — input has {input_count}, alpha has {alpha_count}")


class FrameReadError(CorridorKeyError):
    """Raised when a frame file cannot be read."""

    def __init__(self, clip_name: str, frame_index: int, path: str) -> None:
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.path = path
        super().__init__(f"Clip '{clip_name}': failed to read frame {frame_index} ({path})")


class WriteFailureError(CorridorKeyError):
    """Raised when cv2.imwrite or similar write operation fails."""

    def __init__(self, clip_name: str, frame_index: int, path: str) -> None:
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.path = path
        super().__init__(f"Clip '{clip_name}': failed to write frame {frame_index} ({path})")


class MaskChannelError(CorridorKeyError):
    """Raised when a mask has unexpected channel count that can't be resolved."""

    def __init__(self, clip_name: str, frame_index: int, channels: int) -> None:
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.channels = channels
        super().__init__(f"Clip '{clip_name}': mask frame {frame_index} has {channels} channels, expected 1 or 3+")


class InvalidStateTransitionError(CorridorKeyError):
    """Raised when a clip state transition is not allowed."""

    def __init__(self, clip_name: str, current_state: str, target_state: str) -> None:
        self.clip_name = clip_name
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(f"Clip '{clip_name}': invalid state transition {current_state} -> {target_state}")
