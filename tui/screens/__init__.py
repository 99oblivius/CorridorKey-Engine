"""CorridorKey TUI screens."""

from __future__ import annotations

from .clip_manager import ClipManagerPanel
from .generate_mattes import GeneratePanel
from .global_settings import GlobalSettingsPanel
from .inference import InferencePanel

__all__ = [
    "ClipManagerPanel",
    "GeneratePanel",
    "GlobalSettingsPanel",
    "InferencePanel",
]
