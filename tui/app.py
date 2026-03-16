"""CorridorKey Textual TUI application."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme
from textual.widgets import ContentSwitcher, Footer, Static

from .screens import (
    ClipManagerPanel,
    GeneratePanel,
    GlobalSettingsPanel,
    InferencePanel,
)
from .client import TUIEngineClient
from .theme import THEME, css_variables
from .widgets.header import CorridorKeyHeader

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Resize

# Minimum terminal dimensions for a usable layout.
MIN_WIDTH = 60
MIN_HEIGHT = 20


class _TooSmallOverlay(Static):
    """Overlay shown when the terminal is below minimum size."""

    DEFAULT_CSS = """
    _TooSmallOverlay {
        width: 100%;
        height: 100%;
        align: center middle;
        background: $background;
        layer: overlay;
    }
    """

    def __init__(self, width: int, height: int) -> None:
        msg = (
            f"Terminal too small ({width}x{height}).\n"
            f"Minimum size: {MIN_WIDTH}x{MIN_HEIGHT}.\n\n"
            "Please resize your terminal."
        )
        super().__init__(msg, id="too-small-overlay")


class _ShuttingDownOverlay(Static):
    """Full-screen message shown while the engine shuts down."""

    DEFAULT_CSS = """
    _ShuttingDownOverlay {
        width: 100%;
        height: 100%;
        content-align: center middle;
        background: $background;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__(
            "Shutting down engine...\n\n"
            "Freeing VRAM and closing processes.",
            id="shutting-down",
        )


class CorridorKeyApp(App):
    """Root Textual application for CorridorKey."""

    TITLE = "CorridorKey Engine"
    CSS_PATH = "styles.tcss"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("1", "goto('clip_manager')", "Clips", show=False, priority=True),
        Binding("2", "goto('generate_mattes')", "Generate", show=False, priority=True),
        Binding("3", "goto('inference')", "Inference", show=False, priority=True),
        Binding("4", "goto('global_settings')", "Settings", show=False, priority=True),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.initial_path = initial_path
        self._active_page: str = "clip_manager"
        self.engine: TUIEngineClient | None = None

    def compose(self) -> ComposeResult:
        yield CorridorKeyHeader(active_mode="clip_manager")
        with ContentSwitcher(initial="clip_manager", id="pages"):
            yield ClipManagerPanel(id="clip_manager")
            yield GeneratePanel(id="generate_mattes")
            yield InferencePanel(id="inference")
            yield GlobalSettingsPanel(id="global_settings")
        yield Footer()

    def on_mount(self) -> None:
        """Register theme."""
        custom_vars = css_variables()
        self.register_theme(
            Theme(
                name="corridorkey",
                primary=THEME.primary,
                secondary=THEME.secondary,
                background=THEME.background,
                surface=THEME.surface,
                panel=THEME.panel,
                warning=THEME.warning,
                error=THEME.error,
                success=THEME.success,
                accent=THEME.secondary,
                variables=custom_vars,
            )
        )
        self.theme = "corridorkey"
        self.engine = TUIEngineClient(self)

    def action_goto(self, mode: str) -> None:
        """Switch visible page and update header."""
        self._active_page = mode
        self.query_one("#pages", ContentSwitcher).current = mode
        header = self.query_one(CorridorKeyHeader)
        header.set_active(mode)
        # Focus the panel so its bindings appear in the footer
        try:
            panel = self.query_one(f"#{mode}")
            self.set_focus(panel)
        except Exception:
            pass
        # Check readiness on panels that need clips — but not while
        # the engine is busy (that would re-enable the Start button).
        if mode in ("generate_mattes", "inference"):
            try:
                panel = self.query_one(f"#{mode}")
                if self.engine is None or not self.engine.is_busy:
                    panel.check_readiness()
            except Exception:
                pass

    def action_quit(self) -> None:
        """Show shutdown message, clean up engine in background, then exit."""
        # Immediately show a shutdown overlay so the user knows we're working.
        try:
            self.query_one("#pages", ContentSwitcher).display = False
            self.query_one(CorridorKeyHeader).display = False
            self.query_one(Footer).display = False
        except Exception:
            pass
        self.mount(_ShuttingDownOverlay())

        # Run the slow teardown (model unload, process kill) in a thread
        # so the overlay stays visible and the event loop keeps painting.
        engine = self.engine
        self.engine = None

        def _teardown() -> None:
            if engine is not None:
                engine.close(shutdown=True)
            # Schedule exit back on the event loop
            try:
                loop = self._loop
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(self.exit)
            except Exception:
                self.exit()

        threading.Thread(target=_teardown, name="shutdown", daemon=True).start()

    def on_resize(self, event: Resize) -> None:
        """Enforce minimum terminal size."""
        self._check_terminal_size()

    def _check_terminal_size(self) -> None:
        """Show or hide the too-small overlay based on current size."""
        w, h = self.size
        overlay = self.query("#too-small-overlay")
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            if not overlay:
                self.mount(_TooSmallOverlay(w, h))
        elif overlay:
            overlay.remove()
