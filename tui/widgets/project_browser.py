"""Project browser widget — path input with Tab completion and recent projects dropdown."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from textual import on
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

_MAX_SUGGESTIONS = 8
_MAX_RECENT = 10


class ProjectSelected(Message):
    """Emitted when the user confirms a valid project path."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path


class ProjectBrowser(Widget):
    """Path entry with Tab completion and a dropdown showing completions + recent projects."""

    DEFAULT_CSS = """
    ProjectBrowser {
        height: auto;
        max-height: 16;
        padding: 0 1;
    }

    ProjectBrowser #path-label {
        height: 1;
        color: $text-muted;
    }

    ProjectBrowser #path-input {
        margin-bottom: 0;
    }

    ProjectBrowser #dropdown {
        height: auto;
        max-height: 12;
        display: none;
        background: $surface;
        border: solid $border;
    }

    ProjectBrowser #dropdown.visible {
        display: block;
    }

    ProjectBrowser #completions-label {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }

    ProjectBrowser #completions-list {
        height: auto;
        max-height: 5;
    }

    ProjectBrowser #recent-label {
        height: auto;
        color: $text-muted;
        padding: 1 1 0 1;
    }

    ProjectBrowser #recent-list {
        height: auto;
        max-height: 5;
    }
    """

    def __init__(
        self,
        recent_projects: list[str] | None = None,
        initial_path: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._recent = list(recent_projects or [])
        self._initial_path = initial_path or os.getcwd()
        self._confirmed_path = self._initial_path
        self._mounted = False
        self._active_list: object = None

    def compose(self) -> ComposeResult:
        yield Label("Project Path  [dim](Tab complete · Enter to load)[/]", id="path-label")
        yield Input(
            value=self._initial_path,
            placeholder="Enter project directory...",
            id="path-input",
        )
        with Widget(id="dropdown"):
            yield Static("Completions", id="completions-label")
            yield ListView(id="completions-list")
            yield Static("Recent Projects", id="recent-label")
            yield ListView(id="recent-list")

    def on_mount(self) -> None:
        """Populate the recent projects list."""
        self._populate_recent()
        self.call_after_refresh(self._enable)

    def _enable(self) -> None:
        self._mounted = True

    def _populate_recent(self) -> None:
        """Fill the recent projects ListView."""
        recent_list = self.query_one("#recent-list", ListView)
        recent_list.clear()
        for p in self._recent[:_MAX_RECENT]:
            recent_list.append(ListItem(Label(p), name=p))

    def set_recent(self, projects: list[str]) -> None:
        """Update the recent projects list."""
        self._recent = list(projects)
        self._populate_recent()

    # -- Input events -------------------------------------------------------

    @on(Input.Submitted, "#path-input")
    def _on_path_submitted(self, event: Input.Submitted) -> None:
        path = os.path.expanduser(event.value.strip())
        self._hide_dropdown()
        if os.path.isdir(path):
            self._confirmed_path = path
            self.app.set_focus(None)
            self.post_message(ProjectSelected(path))

    def _on_descendant_focus(self, _event: object) -> None:
        """Show dropdown when the input gets focus."""
        inp = self.query_one("#path-input", Input)
        if inp.has_focus and self._mounted:
            self._update_completions(inp.value)
            self._show_dropdown()

    def _on_descendant_blur(self, _event: object) -> None:
        """Hide dropdown after a short delay (allows clicks to register)."""
        self.call_after_refresh(self._check_blur)

    def _check_blur(self) -> None:
        inp = self.query_one("#path-input", Input)
        comp_list = self.query_one("#completions-list", ListView)
        rec_list = self.query_one("#recent-list", ListView)
        if not inp.has_focus and not comp_list.has_focus and not rec_list.has_focus:
            self._hide_dropdown()

    # -- Completions --------------------------------------------------------

    @on(ListView.Selected, "#completions-list")
    def _on_completion_selected(self, event: ListView.Selected) -> None:
        if event.item.name:
            inp = self.query_one("#path-input", Input)
            inp.value = event.item.name
            inp.cursor_position = len(inp.value)
            self._update_completions(inp.value)

    @on(ListView.Selected, "#recent-list")
    def _on_recent_selected(self, event: ListView.Selected) -> None:
        if event.item.name:
            inp = self.query_one("#path-input", Input)
            inp.value = event.item.name
            inp.cursor_position = len(inp.value)
            self._hide_dropdown()
            if os.path.isdir(event.item.name):
                self._confirmed_path = event.item.name
                self.app.set_focus(None)
                self.post_message(ProjectSelected(event.item.name))

    def on_key(self, event: object) -> None:
        """Handle Tab, arrows, Escape when the path input has focus."""
        key = getattr(event, "key", None)
        inp = self.query_one("#path-input", Input)
        if not inp.has_focus:
            return

        if key == "tab":
            event.prevent_default()
            event.stop()
            self._complete_path()
        elif key == "escape":
            event.prevent_default()
            event.stop()
            inp.value = self._confirmed_path
            inp.cursor_position = len(inp.value)
            self._hide_dropdown()
            self.app.set_focus(None)
        elif key in ("down", "up"):
            event.prevent_default()
            event.stop()
            self._arrow_navigate(key)

    def _arrow_navigate(self, direction: str) -> None:
        """Move highlight through completions then recent items with arrow keys."""
        self._show_dropdown()
        comp_list = self.query_one("#completions-list", ListView)
        rec_list = self.query_one("#recent-list", ListView)

        # Build a flat list of (listview, index) pairs
        items: list[tuple[ListView, int]] = []
        for lv in (comp_list, rec_list):
            for i in range(len(lv)):
                items.append((lv, i))
        if not items:
            return

        # Find current position
        current = -1
        for idx, (lv, i) in enumerate(items):
            if lv.index == i and lv.has_focus:
                current = idx
                break
            # Also check highlighted state without focus
            if lv.index == i and lv == self._active_list:
                current = idx
                break

        if direction == "down":
            nxt = current + 1 if current < len(items) - 1 else 0
        else:
            nxt = current - 1 if current > 0 else len(items) - 1

        target_lv, target_idx = items[nxt]
        self._active_list = target_lv
        target_lv.index = target_idx

        # Update the input field to show the highlighted path
        try:
            item = target_lv.highlighted_child
            if item is not None and item.name:
                inp = self.query_one("#path-input", Input)
                inp.value = item.name
                inp.cursor_position = len(inp.value)
        except (IndexError, AttributeError):
            pass

    def _complete_path(self) -> None:
        """Tab-complete the current path to the longest common prefix."""
        inp = self.query_one("#path-input", Input)
        text = inp.value.strip()
        if not text:
            return

        completions = self._get_completions(text)
        if not completions:
            return
        if len(completions) == 1:
            result = completions[0]
            if os.path.isdir(result) and not result.endswith(os.sep):
                result += os.sep
            inp.value = result
            inp.cursor_position = len(inp.value)
        else:
            prefix = os.path.commonprefix(completions)
            if prefix and len(prefix) > len(text):
                inp.value = prefix
                inp.cursor_position = len(inp.value)
        self._update_completions(inp.value)
        self._show_dropdown()

    def _update_completions(self, text: str) -> None:
        """Update the completions list based on current text."""
        comp_list = self.query_one("#completions-list", ListView)
        comp_list.clear()
        completions = self._get_completions(text)
        for c in completions[:_MAX_SUGGESTIONS]:
            comp_list.append(ListItem(Label(c), name=c))

    # -- Dropdown visibility ------------------------------------------------

    def _show_dropdown(self) -> None:
        self.query_one("#dropdown").add_class("visible")

    def _hide_dropdown(self) -> None:
        self.query_one("#dropdown").remove_class("visible")

    # -- Filesystem ---------------------------------------------------------

    @staticmethod
    def _get_completions(text: str) -> list[str]:
        """Return filesystem directory completions for the given text.

        Only called on Tab or focus — never on keystroke.
        """
        if not text:
            return []
        text = os.path.expanduser(text)
        if os.path.isdir(text):
            parent = text
            prefix = ""
        else:
            parent = os.path.dirname(text)
            prefix = os.path.basename(text)
        if not os.path.isdir(parent):
            return []
        try:
            entries = os.listdir(parent)
        except PermissionError:
            return []
        matches = []
        for entry in sorted(entries):
            if entry.startswith("."):
                continue
            full = os.path.join(parent, entry)
            if os.path.isdir(full) and entry.lower().startswith(prefix.lower()):
                matches.append(full)
        return matches
