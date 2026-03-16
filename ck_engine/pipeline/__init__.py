"""Pipeline operations for CorridorKey.

This package provides alpha generation and inference pipeline functions.
All public symbols are re-exported here for backwards compatibility::

    from ck_engine.pipeline import generate_alpha_hints, run_inference
"""

# Import resolve_device first so submodules can reference it through the
# package namespace (important for mock.patch("ck_engine.pipeline.resolve_device")
# compatibility in tests).
from ck_engine.device import resolve_device

from ck_engine.pipeline.generate import AlphaMode, generate_alpha_hints
from ck_engine.pipeline.inference import InferenceSettings, run_inference

__all__ = [
    "AlphaMode",
    "InferenceSettings",
    "generate_alpha_hints",
    "resolve_device",
    "run_inference",
]
