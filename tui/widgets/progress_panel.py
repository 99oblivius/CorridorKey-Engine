"""Progress panel widget — per-clip progress bars with speed and ETA."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ClipProgress(Widget):
    """Progress display for a single clip."""

    DEFAULT_CSS = """
    ClipProgress {
        height: 3;
        padding: 0 1;
        width: 100%;
        layout: vertical;
    }

    ClipProgress .progress-header {
        height: 1;
        width: 100%;
        layout: horizontal;
    }

    ClipProgress .clip-label {
        width: 1fr;
        text-style: bold;
    }

    ClipProgress .io-label {
        width: auto;
        color: $text-muted;
        margin-right: 2;
    }

    ClipProgress .speed-label {
        width: auto;
        color: $text-muted;
    }

    ClipProgress .eta-label {
        width: auto;
        color: $text-muted;
        margin-left: 2;
    }

    ClipProgress ProgressBar {
        width: 100%;
    }

    ClipProgress Bar {
        width: 1fr;
    }
    """

    done: reactive[int] = reactive(0)
    total: reactive[int] = reactive(0)

    def __init__(self, clip_name: str, total_frames: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._clip_label = clip_name  # display name (may contain spaces etc.)
        self._clip_name = ProgressPanel._sanitize_id(clip_name)  # safe for widget IDs
        self.total = total_frames
        self._start_time: float | None = None

    def compose(self) -> ComposeResult:
        with Widget(classes="progress-header"):
            yield Static(self._clip_label, classes="clip-label")
            yield Static("", classes="io-label", id=f"io-{self._clip_name}")
            yield Static("", classes="speed-label", id=f"speed-{self._clip_name}")
            yield Static("", classes="eta-label", id=f"eta-{self._clip_name}")
        yield ProgressBar(total=self.total, show_percentage=True, id=f"bar-{self._clip_name}")

    def update_progress(
        self, done: int, total: int,
        read_speed: str = "", write_speed: str = "",
    ) -> None:
        """Update the progress bar, IO speeds, frame rate, and ETA."""
        if self._start_time is None and done > 0:
            self._start_time = time.monotonic()

        self.done = done
        self.total = total

        bar = self.query_one(f"#bar-{self._clip_name}", ProgressBar)
        bar.update(total=total, progress=done)

        # IO speeds
        io_label = self.query_one(f"#io-{self._clip_name}", Static)
        io_parts = []
        if read_speed:
            io_parts.append(f"R {read_speed}")
        if write_speed:
            io_parts.append(f"W {write_speed}")
        io_label.update(" | ".join(io_parts))

        # Frame rate and ETA
        speed_label = self.query_one(f"#speed-{self._clip_name}", Static)
        eta_label = self.query_one(f"#eta-{self._clip_name}", Static)

        if self._start_time and done > 0:
            elapsed = time.monotonic() - self._start_time
            fps = done / elapsed if elapsed > 0 else 0
            speed_label.update(f"{fps:.1f} fps")

            remaining = total - done
            if fps > 0:
                eta_secs = remaining / fps
                mins, secs = divmod(int(eta_secs), 60)
                eta_label.update(f"ETA {mins}:{secs:02d}")
            else:
                eta_label.update("")
        else:
            speed_label.update("")
            eta_label.update("")


class QueuedClip(Static):
    """Placeholder for a clip waiting in the queue."""

    DEFAULT_CSS = """
    QueuedClip {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, clip_name: str, frame_count: int, alpha_frames: int = 0, **kwargs: object) -> None:
        if alpha_frames > 0:
            text = f"  {clip_name}  queued ({frame_count} input, {alpha_frames} matte frames)"
        else:
            text = f"  {clip_name}  queued ({frame_count} input frames, no mattes)"
        super().__init__(text, **kwargs)


class ProgressPanel(Widget):
    """Panel showing progress for active and queued clips."""

    DEFAULT_CSS = """
    ProgressPanel {
        height: auto;
        max-height: 20;
        width: 100%;
    }

    ProgressPanel #progress-container {
        height: auto;
        width: 100%;
    }
    """

    _epoch: int = 0  # monotonic counter to avoid widget ID collisions

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        # Buffer progress updates that arrive before the widget is mounted.
        self._pending_progress: dict[str, tuple[int, int]] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(id="progress-container")

    @staticmethod
    def _sanitize_id(name: str) -> str:
        """Sanitize a name for use in a Textual widget ID.

        Textual IDs may only contain letters, numbers, underscores, and
        hyphens.  Paths with spaces, parentheses, or other special
        characters (e.g. ``New folder (4)``) must be cleaned.
        """
        import re
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    def _tag(self, name: str) -> str:
        """Return a unique ID fragment for *name* in the current epoch."""
        return f"{self._sanitize_id(name)}-{self._epoch}"

    def set_queue(self, clips: list[tuple[str, int]]) -> None:
        """Set the initial queue of clips to process.

        *clips* is a list of ``(name, frame_count)`` tuples.
        """
        self._epoch += 1
        container = self.query_one("#progress-container", Vertical)
        container.remove_children()
        for name, count in clips:
            container.mount(QueuedClip(name, count, id=f"queued-{self._tag(name)}"))

    def set_queue_with_info(self, clips: list[tuple[str, int, int]]) -> None:
        """Set the initial queue with input and matte frame info.

        *clips* is a list of ``(name, input_frame_count, alpha_frame_count)`` tuples.
        """
        self._epoch += 1
        container = self.query_one("#progress-container", Vertical)
        container.remove_children()
        for name, input_count, alpha_count in clips:
            container.mount(
                QueuedClip(name, input_count, alpha_frames=alpha_count, id=f"queued-{self._tag(name)}")
            )

    def activate_clip(self, clip_name: str, total_frames: int) -> None:
        """Promote a queued clip to active with a progress bar."""
        # Bump epoch so new widgets never collide with stale async removals.
        self._epoch += 1
        container = self.query_one("#progress-container", Vertical)
        # Clear any leftover widgets from previous runs.
        container.remove_children()
        # Add progress widget with the new epoch-tagged ID.
        widget = ClipProgress(clip_name, total_frames, id=f"progress-{self._tag(clip_name)}")
        container.mount(widget, before=0)

        # Apply any buffered progress that arrived before the widget existed.
        pending = self._pending_progress.pop(clip_name, None)
        if pending:
            widget.update_progress(*pending)

    def update_progress(
        self, clip_name: str, done: int, total: int,
        read_speed: str = "", write_speed: str = "",
    ) -> None:
        """Update the active clip's progress bar.

        If the widget hasn't been mounted yet, buffer the latest value
        so ``activate_clip`` can apply it once the widget exists.
        """
        widgets = self.query(f"#progress-{self._tag(clip_name)}")
        if widgets:
            widgets.first(ClipProgress).update_progress(
                done, total, read_speed=read_speed, write_speed=write_speed,
            )
        else:
            self._pending_progress[clip_name] = (done, total)

    def clear(self) -> None:
        """Remove all progress entries."""
        container = self.query_one("#progress-container", Vertical)
        container.remove_children()
