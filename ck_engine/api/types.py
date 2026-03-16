"""Public API types for the CorridorKey engine protocol.

All types are frozen dataclasses with ``to_dict()`` / ``from_dict()``
for JSON-RPC serialization.  These form the contract between engine
and frontends -- keep them stable across releases.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ck_engine.config import (
    DEFAULT_DESPECKLE_SIZE,
    DEFAULT_DESPILL_STRENGTH,
    DEFAULT_IMG_SIZE,
    DEFAULT_REFINER_SCALE,
)

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "original": {
        "flash_attention": False,
        "tiled_refiner": False,
        "disable_cudnn_benchmark": False,
        "cache_clearing": False,
        "mixed_precision": False,
        "model_precision": "float32",
        "gpu_postprocess": True,
        "comp_format": "exr",
        "comp_checkerboard": False,
        "dma_buffers": 2,
        "compile_mode": "none",
        "cuda_graphs": False,
        "tensorrt": False,
        "token_routing": False,
    },
    "optimized": {
        "flash_attention": True,
        "tiled_refiner": True,
        "disable_cudnn_benchmark": True,
        "cache_clearing": True,
        "mixed_precision": True,
        "model_precision": "float16",
        "gpu_postprocess": True,
        "comp_format": "exr",
        "comp_checkerboard": False,
        "dma_buffers": 2,
        "compile_mode": "none",
        "cuda_graphs": False,
        "tensorrt": False,
        "token_routing": False,
    },
    "performance": {
        "flash_attention": True,
        "tiled_refiner": False,
        "disable_cudnn_benchmark": False,
        "cache_clearing": False,
        "mixed_precision": True,
        "model_precision": "float16",
        "gpu_postprocess": True,
        "comp_format": "exr",
        "comp_checkerboard": False,
        "dma_buffers": 3,
        "compile_mode": "max-autotune",
        "cuda_graphs": False,
        "tensorrt": False,
        "token_routing": False,
    },
    "experimental": {
        "flash_attention": True,
        "tiled_refiner": True,
        "disable_cudnn_benchmark": True,
        "cache_clearing": True,
        "mixed_precision": True,
        "model_precision": "float16",
        "gpu_postprocess": True,
        "comp_format": "exr",
        "comp_checkerboard": False,
        "dma_buffers": 2,
        "compile_mode": "default",
        "cuda_graphs": False,
        "tensorrt": False,
        "token_routing": True,
    },
}


def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or list/dict of dataclasses) to a plain dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = _dc_to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, list):
        return [_dc_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    return obj


def _dc_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Construct a frozen dataclass from *data*, ignoring unknown keys.

    Handles nested dataclass fields by inspecting the field type annotation.
    """
    if not isinstance(data, dict):
        return data

    known_fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in known_fields:
            continue
        f = known_fields[key]
        field_type = _resolve_field_type(cls, f)

        if value is not None and field_type is not None and dataclasses.is_dataclass(field_type):
            kwargs[key] = _dc_from_dict(field_type, value)
        elif value is not None and isinstance(value, list) and field_type is not None:
            # Handle list[SomeDataclass]
            inner = _resolve_list_inner(cls, f)
            if inner is not None and dataclasses.is_dataclass(inner):
                kwargs[key] = [_dc_from_dict(inner, item) if isinstance(item, dict) else item for item in value]
            else:
                kwargs[key] = value
        elif value is not None and isinstance(value, dict) and field_type is not None:
            # Handle dict[str, SomeDataclass | None]
            inner = _resolve_dict_value_type(cls, f)
            if inner is not None and dataclasses.is_dataclass(inner):
                kwargs[key] = {
                    k: _dc_from_dict(inner, v) if isinstance(v, dict) else v for k, v in value.items()
                }
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _resolve_field_type(cls: type, f: dataclasses.Field) -> type | None:  # type: ignore[type-arg]
    """Resolve a field's type annotation to a concrete type.

    For ``Optional[X]`` returns ``X`` (or its origin if generic).
    For ``list[X]`` returns ``list``.  For ``dict[K,V]`` returns ``dict``.
    For plain types returns the type itself.
    """
    import typing

    hints = typing.get_type_hints(cls)
    hint = hints.get(f.name)
    if hint is None:
        return None
    # Try unwrapping Optional first
    unwrapped = _unwrap_optional(hint)
    if unwrapped is not None:
        if isinstance(unwrapped, type):
            return unwrapped
        # Generic alias like list[str] -- return origin (list, dict, etc.)
        origin = getattr(unwrapped, "__origin__", None)
        if origin is not None and isinstance(origin, type):
            return origin
        return None
    # For generics like list[X], dict[K,V] return the origin
    origin = getattr(hint, "__origin__", None)
    if origin is not None and isinstance(origin, type):
        return origin
    return None


def _unwrap_optional(hint: Any) -> Any:
    """Unwrap Optional[X] / X | None to X.

    Returns the inner type if *hint* is ``Optional[X]``, ``Union[X, None]``,
    or ``X | None``.  Returns *hint* unchanged if it is already a plain type.
    Returns ``None`` if the hint cannot be resolved.

    Note: the return may be a generic alias (e.g. ``list[str]``) rather
    than a plain ``type``.
    """
    import types as _types
    import typing

    # Plain concrete type -- nothing to unwrap
    if isinstance(hint, type):
        return hint

    origin = getattr(hint, "__origin__", None)

    # typing.Union or types.UnionType (X | Y syntax on 3.10+)
    is_union = origin is typing.Union or isinstance(hint, _types.UnionType)
    if is_union:
        args = getattr(hint, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        return None

    # Not a union -- could be list[X], dict[K,V], etc.  Return None to
    # signal that this is not a simple unwrappable type.
    return None


def _resolve_list_inner(cls: type, f: dataclasses.Field) -> type | None:  # type: ignore[type-arg]
    """For ``list[X]`` or ``list[X] | None``, return ``X``."""
    import typing

    hints = typing.get_type_hints(cls)
    hint = hints.get(f.name)
    if hint is None:
        return None
    # Unwrap Optional[list[X]] -> list[X]
    unwrapped = _unwrap_optional(hint)
    if unwrapped is not None:
        if isinstance(unwrapped, type):
            # unwrapped to a plain type (e.g. str) -- not a list generic
            return None
        # unwrapped to a generic alias (e.g. list[str]) -- use it
        hint = unwrapped
    # hint is now list[X] (or the original if not Optional)
    origin = getattr(hint, "__origin__", None)
    if origin is not list:
        return None
    args = getattr(hint, "__args__", None)
    if args and len(args) >= 1:
        t = args[0]
        return t if isinstance(t, type) else None
    return None


def _resolve_dict_value_type(cls: type, f: dataclasses.Field) -> type | None:  # type: ignore[type-arg]
    """For ``dict[K, V]`` or ``dict[K, V | None]``, return the unwrapped ``V``."""
    import typing

    hints = typing.get_type_hints(cls)
    hint = hints.get(f.name)
    if hint is None:
        return None
    # Unwrap Optional[dict[K,V]] -> dict[K,V]
    unwrapped = _unwrap_optional(hint)
    if unwrapped is not None:
        if isinstance(unwrapped, type):
            return None
        hint = unwrapped
    origin = getattr(hint, "__origin__", None)
    if origin is not dict:
        return None
    args = getattr(hint, "__args__", None)
    if args and len(args) >= 2:
        val_type = _unwrap_optional(args[1])
        return val_type if isinstance(val_type, type) else None
    return None


# ---------------------------------------------------------------------------
# API Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetInfo:
    """Describes a clip's input/alpha/mask asset."""

    type: str  # "sequence" or "video"
    frame_count: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class ClipInfo:
    """Describes a discovered clip (serializable mirror of ClipEntry)."""

    name: str
    root_path: str
    state: str  # ClipState enum value as string (e.g. "RAW", "READY")
    input: AssetInfo | None = None
    alpha: AssetInfo | None = None
    mask: AssetInfo | None = None
    has_outputs: bool = False
    completed_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class InferenceSettings:
    """Per-job inference parameters (API-side frozen copy).

    Mirrors ``backend.pipeline.InferenceSettings`` but is frozen for
    safe transport across the API boundary.
    """

    input_is_linear: bool = False
    despill_strength: float = DEFAULT_DESPILL_STRENGTH
    auto_despeckle: bool = True
    despeckle_size: int = DEFAULT_DESPECKLE_SIZE
    refiner_scale: float = DEFAULT_REFINER_SCALE

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceSettings:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class OptimizationParams:
    """Optimization configuration for inference jobs.

    Fields set to ``None`` mean "use profile defaults" (if a profile is
    given) or "use engine defaults".  Only non-None fields override.
    """

    profile: str | None = None
    flash_attention: bool | None = None
    tiled_refiner: bool | None = None
    tile_size: int | None = None
    tile_overlap: int | None = None
    cache_clearing: bool | None = None
    disable_cudnn_benchmark: bool | None = None
    compile_mode: str | None = None
    model_precision: str | None = None
    gpu_postprocess: bool | None = None
    comp_format: str | None = None
    comp_checkerboard: bool | None = None
    dma_buffers: int | None = None
    cuda_graphs: bool | None = None
    tensorrt: bool | None = None
    mixed_precision: bool | None = None
    token_routing: bool | None = None

    def resolve(self) -> dict[str, Any]:
        """Produce a dict of only the effective non-None fields.

        If a *profile* is set, its defaults are used as the base and any
        explicit non-None fields override them.  If no profile is set,
        only the explicitly provided fields are returned.
        """
        base: dict[str, Any] = {}
        if self.profile is not None:
            if self.profile not in _PROFILE_DEFAULTS:
                raise ValueError(
                    f"Unknown optimization profile '{self.profile}'. "
                    f"Valid profiles: {', '.join(sorted(_PROFILE_DEFAULTS))}"
                )
            base = dict(_PROFILE_DEFAULTS[self.profile])

        # Override with any explicit (non-None) fields, skipping 'profile'
        for f in dataclasses.fields(self):
            if f.name == "profile":
                continue
            value = getattr(self, f.name)
            if value is not None:
                base[f.name] = value

        return base

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationParams:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class GenerateParams:
    """Job parameters for alpha generation."""

    path: str
    model: str = "birefnet"
    mode: str = "replace"
    frames: str | None = None
    device: str = "auto"
    halt_on_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateParams:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class InferenceParams:
    """Job parameters for inference."""

    path: str
    frames: str | None = None
    device: str = "auto"
    backend: str = "auto"
    settings: InferenceSettings = field(default_factory=InferenceSettings)
    optimization: OptimizationParams | None = None
    devices: list[str] | None = None
    img_size: int = DEFAULT_IMG_SIZE
    read_workers: int = 0
    write_workers: int = 0
    cpus: int = 0
    gpu_resilience: bool = False
    halt_on_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceParams:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class DeviceInfo:
    """GPU device descriptor."""

    id: str  # e.g. "cuda:0"
    name: str  # e.g. "RTX 4090"
    vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class VRAMInfo:
    """VRAM state."""

    total_mb: float
    used_mb: float
    free_mb: float

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VRAMInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class LoadedModelInfo:
    """Info about a loaded model."""

    backend: str
    device: str
    vram_mb: float
    config_hash: str = ""
    img_size: int = 0
    precision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoadedModelInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class EngineCapabilities:
    """Response to engine.capabilities."""

    version: str
    generators: list[str]
    backends: list[str]
    devices: list[DeviceInfo]
    profiles: list[str]
    transport: str

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineCapabilities:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class EngineStatus:
    """Response to engine.status."""

    state: str  # "idle", "busy", "shutting_down"
    active_job: str | None = None
    models_loaded: dict[str, LoadedModelInfo | None] = field(default_factory=dict)
    vram: VRAMInfo | None = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineStatus:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class JobStatus:
    """Response to job.status."""

    job_id: str
    state: str  # "running", "completed", "failed", "cancelled"
    type: str  # "generate" or "inference"
    current_clip: str
    progress: dict[str, int] = field(default_factory=dict)  # {"done": N, "total": N}
    clips_completed: int = 0
    clips_total: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatus:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class FailedFrameInfo:
    """Detail about a failed frame."""

    clip: str
    frame: int
    error: str

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailedFrameInfo:
        return _dc_from_dict(cls, data)


@dataclass(frozen=True)
class JobResult:
    """Returned from job.submit on acceptance."""

    job_id: str
    clips: list[ClipInfo]
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        return _dc_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobResult:
        return _dc_from_dict(cls, data)
