"""Backwards-compatible shim — settings moved to backend.settings."""
from ck_engine.settings import *  # noqa: F401,F403
from ck_engine.settings import GlobalSettings, ProjectSettings  # explicit re-export

__all__ = ["GlobalSettings", "ProjectSettings"]
