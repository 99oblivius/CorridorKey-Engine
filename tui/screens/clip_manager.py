"""Clip Manager panel — project browser with DirectoryTree."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.containers import Vertical
from textual.widgets import Static

from ..widgets.clip_tree import ClipTree, SelectionChanged
from ..widgets.project_browser import ProjectBrowser, ProjectSelected

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ClipManagerPanel(Vertical):
    """Home panel: project path browser and clip tree."""

    DEFAULT_CSS = """
    ClipManagerPanel {
        layout: vertical;
    }

    ClipManagerPanel #summary-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        initial = getattr(self.app, "initial_path", None) or os.getcwd()
        recent = self._load_recent()
        yield ProjectBrowser(
            recent_projects=recent,
            initial_path=initial,
            id="project-browser",
        )
        yield ClipTree(initial, id="clip-tree")
        yield Static("No clips loaded", id="summary-bar")

    def on_mount(self) -> None:
        """If launched with an initial path, trigger scan."""
        initial = getattr(self.app, "initial_path", None)
        if initial and os.path.isdir(initial):
            self._update_summary()

    @on(ProjectSelected)
    def _on_project_selected(self, event: ProjectSelected) -> None:
        """Load a project when the browser confirms a path."""
        tree = self.query_one("#clip-tree", ClipTree)
        tree.set_path(event.path)
        self._update_summary()

    @on(SelectionChanged)
    def _on_selection_changed(self, event: SelectionChanged) -> None:
        """Persist selection and update summary bar."""
        from tui.selection_io import SelectionMap

        path = self._get_current_path()
        if path:
            sel = SelectionMap(clips=event.clips)
            sel.save(Path(path))

        self._update_summary()

    def _update_summary(self) -> None:
        """Update the summary bar with clip/frame counts."""
        try:
            tree = self.query_one("#clip-tree", ClipTree)
            total, selected, frames, sel_frames = tree.get_summary()
            summary = self.query_one("#summary-bar", Static)
            if total == 0:
                summary.update("No clips loaded")
            else:
                summary.update(
                    f"{total} clips | {selected} selected | "
                    f"{frames} frames | {sel_frames} selected frames"
                )
        except Exception:
            pass

    def _get_current_path(self) -> str | None:
        """Get the current project path from the browser input."""
        try:
            from textual.widgets import Input

            inp = self.query_one("#path-input", Input)
            path = inp.value.strip()
            return path if path and os.path.isdir(path) else None
        except Exception:
            return None

    def refresh_recent(self) -> None:
        """Reload the recent projects list from settings."""
        browser = self.query_one("#project-browser", ProjectBrowser)
        browser.set_recent(self._load_recent())

    @staticmethod
    def _load_recent() -> list[str]:
        """Load recent projects from global settings."""
        try:
            from ck_engine.settings import GlobalSettings

            return GlobalSettings.load().recent_projects
        except Exception:
            return []
