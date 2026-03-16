"""CorridorKey theme system.

All colors live in :class:`CorridorKeyTheme` — edit the dataclass fields
to retheme the entire application.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from textual.design import ColorSystem


@dataclass
class CorridorKeyTheme:
    """All colors in one place.  Edit these to retheme the entire app."""

    # Brand
    primary: str = "#FFD700"  # Corridor Digital bright yellow
    secondary: str = "#00FF41"  # Green screen bright green

    # Surfaces
    background: str = "#0D0F12"  # Near-black (not pure black)
    surface: str = "#161A20"  # Panel / widget backgrounds
    panel: str = "#1C2028"  # Elevated surfaces
    border: str = "#252C35"  # Subtle borders

    # Text
    text: str = "#E0E4EB"  # Primary text (off-white)
    text_muted: str = "#6B7280"  # Secondary labels
    text_disabled: str = "#383E47"  # Grayed out

    # Semantic
    success: str = "#22C55E"  # COMPLETE state
    warning: str = "#F59E0B"  # Warnings, partial states
    error: str = "#EF4444"  # ERROR state
    processing: str = "#A78BFA"  # Active GPU work (purple)

    # Clip states
    state_raw: str = "#6B7280"  # RAW — muted gray
    state_masked: str = "#F59E0B"  # MASKED — amber
    state_ready: str = "#FFD700"  # READY — yellow (brand)
    state_complete: str = "#22C55E"  # COMPLETE — green
    state_error: str = "#EF4444"  # ERROR — red
    state_running: str = "#A78BFA"  # Processing — purple


# Singleton default theme
THEME = CorridorKeyTheme()


def build_color_system(theme: CorridorKeyTheme | None = None) -> ColorSystem:
    """Convert a :class:`CorridorKeyTheme` into a Textual :class:`ColorSystem`."""
    t = theme or THEME
    return ColorSystem(
        primary=t.primary,
        secondary=t.secondary,
        background=t.background,
        surface=t.surface,
        panel=t.panel,
        warning=t.warning,
        error=t.error,
        success=t.success,
        accent=t.secondary,
    )


def css_variables(theme: CorridorKeyTheme | None = None) -> dict[str, str]:
    """Return a mapping of ``$name`` → color value for use in TCSS."""
    t = theme or THEME
    return {f.name.replace("_", "-"): getattr(t, f.name) for f in fields(t)}
