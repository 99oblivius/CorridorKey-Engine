"""Alpha generator registry — lazy-import wrapper constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AlphaGenerator


def get_generator(name: str, device: str = "cpu") -> AlphaGenerator:
    """Lazy-import and construct a generator by name."""
    if name == "gvm":
        from .gvm.wrapper import GVMAlphaGenerator

        return GVMAlphaGenerator(device=device)
    elif name == "birefnet":
        from .birefnet.wrapper import BiRefNetAlphaGenerator

        return BiRefNetAlphaGenerator(device=device)
    elif name == "videomama":
        from .videomama.wrapper import VideoMaMaAlphaGenerator

        return VideoMaMaAlphaGenerator(device=device)
    else:
        raise ValueError(f"Unknown alpha generator: {name!r}. Choose: gvm, birefnet, videomama")
