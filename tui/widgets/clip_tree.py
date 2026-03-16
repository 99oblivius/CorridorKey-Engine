"""Clip tree widget — DirectoryTree-based project browser with selection and status badges."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

from rich.text import Text
from textual import work
from textual.message import Message
from textual.widgets import DirectoryTree

if TYPE_CHECKING:
    from pathlib import Path

    from rich.style import Style
    from textual.widgets._directory_tree import DirEntry
    from textual.widgets._tree import TreeNode

    from ck_engine.clip_state import ClipEntry

# State -> (display label, Rich style)
_STATE_STYLES: dict[str, tuple[str, str]] = {
    "RAW": ("RAW", "dim"),
    "MASKED": ("MASKED", "yellow"),
    "READY": ("READY", "bold yellow"),
    "COMPLETE": ("DONE", "bold green"),
    "ERROR": ("ERR", "bold red"),
}

# Directories/files to hide from the tree
_HIDDEN: set[str] = {"__pycache__", ".git", ".DS_Store", ".corridorkey_manifest.json"}


class SelectionChanged(Message):
    """Emitted when clip selection changes."""

    def __init__(self, clips: dict[str, list[int] | None]) -> None:
        super().__init__()
        self.clips = clips


class ClipTree(DirectoryTree):
    """Project directory tree with clip status badges and selection checkboxes."""

    show_root = False

    DEFAULT_CSS = """
    ClipTree {
        height: 1fr;
    }
    """

    def __init__(self, path: str | Path = ".", **kwargs: object) -> None:
        super().__init__(str(path), **kwargs)
        self._selected_clips: set[str] = set()
        self._clip_cache: dict[str, ClipEntry] = {}
        self._clip_dirs: set[str] = set()
        self._initialized = False

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        """Hide dotfiles, __pycache__, etc."""
        return [p for p in paths if p.name not in _HIDDEN and not p.name.startswith(".")]

    def render_label(
        self, node: TreeNode[DirEntry], base_style: Style, style: Style
    ) -> Text:
        """Custom label rendering — adds checkboxes and status badges for clip directories."""
        node_data = node.data
        if node_data is not None:
            path_str = str(node_data.path)

            if path_str in self._clip_dirs:
                clip_name = os.path.basename(path_str)
                check = "[x]" if clip_name in self._selected_clips else "[ ]"

                entry = self._clip_cache.get(path_str)
                if entry is not None:
                    state_val = entry.state.value
                    badge_label, badge_style = _STATE_STYLES.get(
                        state_val, ("?", "dim")
                    )
                    frames = (
                        entry.input_asset.frame_count if entry.input_asset else 0
                    )
                    return Text.assemble(
                        (f"{check} ", ""),
                        (f"{clip_name}  ", ""),
                        (f"{badge_label:<8s}", badge_style),
                        (f"  {frames}f", "dim"),
                    )
                return Text.assemble(
                    (f"{check} ", ""),
                    (f"{clip_name}", ""),
                )

        # Fall back to default DirectoryTree rendering
        return super().render_label(node, base_style, style)

    def on_mount(self) -> None:
        """Scan for clip directories once mounted."""
        self._scan_clips()

    def on_tree_node_expanded(self, event: DirectoryTree.NodeExpanded) -> None:
        """When a clip node is expanded, scan its assets if not cached."""
        node_data = event.node.data
        if node_data is not None:
            path_str = str(node_data.path)
            if path_str in self._clip_dirs and path_str not in self._clip_cache:
                self._scan_clip_entry(path_str)

    def on_key(self, event: object) -> None:
        """Handle Space to toggle selection on clip nodes."""
        if getattr(event, "key", None) != "space":
            return

        node = self.cursor_node
        if node is None or node.data is None:
            return

        path_str = str(node.data.path)
        if path_str not in self._clip_dirs:
            return

        event.prevent_default()
        event.stop()
        clip_name = os.path.basename(path_str)
        if clip_name in self._selected_clips:
            self._selected_clips.discard(clip_name)
        else:
            self._selected_clips.add(clip_name)

        # Force re-render of the node label
        node.refresh()
        self._emit_selection()

    def on_tree_node_selected(self, event: DirectoryTree.NodeSelected) -> None:
        """Toggle selection when a clip node is clicked or Enter is pressed."""
        node = event.node
        if node is None or node.data is None:
            return

        path_str = str(node.data.path)
        if path_str not in self._clip_dirs:
            return

        clip_name = os.path.basename(path_str)
        if clip_name in self._selected_clips:
            self._selected_clips.discard(clip_name)
        else:
            self._selected_clips.add(clip_name)

        node.refresh()
        self._emit_selection()

    @work(thread=True)
    def _scan_clips(self) -> None:
        """Scan the root path for clip directories (ones with Input/)."""
        from ck_engine.clip_state import ClipEntry
        from ck_engine.errors import ClipScanError
        from ck_engine.project import get_clip_dirs

        root = str(self.path)
        if not os.path.isdir(root):
            return

        clip_dirs = get_clip_dirs(root)

        entries: dict[str, ClipEntry] = {}
        found_dirs: set[str] = set()
        names: set[str] = set()

        for clip_dir in clip_dirs:
            name = os.path.basename(clip_dir.rstrip(os.sep))
            if not name:
                continue
            found_dirs.add(clip_dir)
            entry = ClipEntry(name=name, root_path=clip_dir)
            with contextlib.suppress(
                ClipScanError, FileNotFoundError, ValueError, OSError
            ):
                entry.find_assets()
            entries[clip_dir] = entry
            names.add(name)

        self.app.call_from_thread(self._apply_scan, entries, found_dirs, names)

    def _apply_scan(
        self,
        entries: dict[str, ClipEntry],
        found_dirs: set[str],
        names: set[str],
    ) -> None:
        """Apply scan results on the main thread."""
        self._clip_cache.update(entries)
        self._clip_dirs = found_dirs
        if not self._initialized:
            self._selected_clips = set(names)
            self._initialized = True
        self.reload()
        # Auto-expand root so user sees contents, not just the root folder
        if self.root and not self.root.is_expanded:
            self.root.expand()
        self._emit_selection()

    @work(thread=True)
    def _scan_clip_entry(self, path_str: str) -> None:
        """Scan a single clip directory for assets."""
        from ck_engine.clip_state import ClipEntry
        from ck_engine.errors import ClipScanError

        name = os.path.basename(path_str.rstrip(os.sep))
        entry = ClipEntry(name=name, root_path=path_str)
        with contextlib.suppress(
            ClipScanError, FileNotFoundError, ValueError, OSError
        ):
            entry.find_assets()
        self.app.call_from_thread(self._apply_clip_entry, path_str, entry)

    def _apply_clip_entry(self, path_str: str, entry: ClipEntry) -> None:
        """Apply a single clip scan result."""
        self._clip_cache[path_str] = entry
        self.reload()

    def _emit_selection(self) -> None:
        """Build and emit the current selection map."""
        selection: dict[str, list[int] | None] = {}
        for entry in self._clip_cache.values():
            if entry.name not in self._selected_clips:
                selection[entry.name] = []
        self.post_message(SelectionChanged(selection))

    def get_selected_clips(self) -> list[ClipEntry]:
        """Return clips that are currently selected."""
        return [
            entry
            for entry in self._clip_cache.values()
            if entry.name in self._selected_clips
        ]

    def set_path(self, path: str) -> None:
        """Change the root path and rescan."""
        self._clip_cache.clear()
        self._clip_dirs.clear()
        self._selected_clips.clear()
        self._initialized = False
        # Setting .path triggers watch_path -> reload() automatically
        self.path = path
        # Defer scan until the path change and reload have settled
        self.call_after_refresh(self._scan_clips)

    def get_summary(self) -> tuple[int, int, int, int]:
        """Return (total_clips, selected_clips, total_frames, selected_frames)."""
        total_clips = len(self._clip_cache)
        selected_clips = sum(
            1 for e in self._clip_cache.values() if e.name in self._selected_clips
        )
        total_frames = sum(
            e.input_asset.frame_count if e.input_asset else 0
            for e in self._clip_cache.values()
        )
        selected_frames = sum(
            e.input_asset.frame_count if e.input_asset else 0
            for e in self._clip_cache.values()
            if e.name in self._selected_clips
        )
        return total_clips, selected_clips, total_frames, selected_frames
