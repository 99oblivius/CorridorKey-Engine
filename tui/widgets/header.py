"""Custom header widget for CorridorKey TUI — brand + tab bar."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

_TABS: list[tuple[str, str, str]] = [
    ("1", "clip_manager", "Clips"),
    ("2", "generate_mattes", "Generate"),
    ("3", "inference", "Inference"),
    ("4", "global_settings", "Settings"),
]


class _Tab(Static):
    """Clickable tab label that switches app page on click."""

    def __init__(self, key: str, mode: str, label: str, **kwargs: object) -> None:
        super().__init__(f" [{key}] {label} ", **kwargs)
        self._mode = mode

    def on_click(self, event: object) -> None:
        self.app.action_goto(self._mode)


class _QuitButton(Static):
    """Clickable quit label in the header."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("[q] Quit", **kwargs)

    def on_click(self, event: object) -> None:
        self.app.action_quit()


class CorridorKeyHeader(Widget):
    """App header: brand row + numbered tab bar."""

    DEFAULT_CSS = """
    CorridorKeyHeader {
        dock: top;
        height: 2;
        background: $surface;
    }

    CorridorKeyHeader .header-row {
        height: 1;
        width: 100%;
    }

    CorridorKeyHeader #header-brand {
        width: auto;
        color: $primary;
        text-style: bold;
        padding: 0 1;
    }

    CorridorKeyHeader #header-right {
        width: 1fr;
        content-align: right middle;
        color: $text-muted;
        padding: 0 1;
    }

    CorridorKeyHeader .tab-bar {
        height: 1;
        width: 100%;
    }

    CorridorKeyHeader .tab {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    CorridorKeyHeader .tab.active {
        color: $primary;
        text-style: bold;
    }

    CorridorKeyHeader #header-hints {
        width: 1fr;
        content-align: right middle;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, active_mode: str = "clip_manager", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._active_mode = active_mode

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Static("CORRIDOR KEY", id="header-brand")
            yield Static("", id="header-right")
        with Horizontal(classes="tab-bar"):
            for key, mode, label in _TABS:
                classes = "tab active" if mode == self._active_mode else "tab"
                yield _Tab(key, mode, label, classes=classes, id=f"tab-{mode}")
            yield _QuitButton(id="header-hints")

    def set_active(self, mode: str) -> None:
        """Update which tab is highlighted."""
        self._active_mode = mode
        for _key, tab_mode, _label in _TABS:
            with contextlib.suppress(Exception):
                tab = self.query_one(f"#tab-{tab_mode}")
                if tab_mode == mode:
                    tab.add_class("active")
                else:
                    tab.remove_class("active")

    def set_right_content(self, text: str) -> None:
        """Update the right-side content (e.g. GPU status text)."""
        with contextlib.suppress(Exception):
            self.query_one("#header-right", Static).update(text)
