"""Settings persistence for CorridorKey TUI.

* :class:`GlobalSettings` — per-installation, stored as TOML.
* :class:`ProjectSettings` — per-project, stored as JSON.
"""

from __future__ import annotations

import json
import logging
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from CorridorKeyModule.constants import (
    DEFAULT_DESPECKLE_SIZE,
    DEFAULT_DESPILL_STRENGTH,
    DEFAULT_IMG_SIZE,
    DEFAULT_REFINER_SCALE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
)

logger = logging.getLogger(__name__)

GLOBAL_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "tools" / "user_settings.toml"

_SETTINGS_VERSION = 1


# ---------------------------------------------------------------------------
# TOML writer (tomllib is read-only; we format manually)
# ---------------------------------------------------------------------------


def _toml_value(v: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v!r}"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        inner = ", ".join(_toml_value(item) for item in v)
        return f"[{inner}]"
    if v is None:
        return '""'  # TOML has no null; use empty string as sentinel
    return f'"{v}"'


def _write_toml(data: dict[str, object], path: Path) -> None:
    """Write *data* as a flat TOML file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"_version = {_SETTINGS_VERSION}", ""]
    for key, val in data.items():
        if key == "_version":
            continue
        lines.append(f"{key} = {_toml_value(val)}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# GlobalSettings
# ---------------------------------------------------------------------------


@dataclass
class GlobalSettings:
    """Per-installation settings.  Persisted to ``tools/user_settings.toml``."""

    # Device
    device: str = "auto"
    backend: str = "auto"
    devices: list[str] = field(default_factory=list)

    # Pipeline
    img_size: int = DEFAULT_IMG_SIZE
    read_workers: int = 0
    write_workers: int = 0
    cpus: int = 0
    gpu_resilience: bool = False

    # Optimization
    profile: str = ""
    flash_attention: bool | None = None
    tiled_refiner: bool | None = None
    cache_clearing: bool | None = None
    disable_cudnn_benchmark: bool | None = None
    compile_mode: str = ""
    precision: str = "fp16"
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    gpu_postprocess: bool | None = None
    comp_format: str = "exr"
    comp_checkerboard: bool = False
    dma_buffers: int = 2

    # Recent projects
    recent_projects: list[str] = field(default_factory=list)

    def save(self, path: Path | None = None) -> None:
        """Write settings to TOML."""
        dest = path or GLOBAL_SETTINGS_PATH
        data = self._to_dict()
        _write_toml(data, dest)

    @classmethod
    def load(cls, path: Path | None = None) -> GlobalSettings:
        """Load settings from TOML, falling back to defaults for missing keys."""
        src = path or GLOBAL_SETTINGS_PATH
        if not src.exists():
            return cls()
        try:
            raw = tomllib.loads(src.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse %s, using defaults", src)
            return cls()
        return cls._from_dict(raw)

    def _to_dict(self) -> dict[str, object]:
        """Serialize to a flat dict suitable for TOML."""
        d: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            # Encode None bools as empty string (TOML has no null)
            if val is None:
                d[f.name] = ""
            else:
                d[f.name] = val
        return d

    @classmethod
    def _from_dict(cls, raw: dict[str, object]) -> GlobalSettings:
        """Deserialize from a parsed TOML dict, handling missing/extra keys."""
        defaults = cls()
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            val = raw[f.name]
            default_val = getattr(defaults, f.name)
            # Decode empty-string sentinel back to None for Optional[bool]
            if default_val is None and val == "":
                kwargs[f.name] = None
            elif isinstance(default_val, bool) and isinstance(val, str):
                kwargs[f.name] = val.lower() in ("true", "1", "yes")
            elif isinstance(default_val, int) and not isinstance(default_val, bool):
                kwargs[f.name] = int(val)
            elif isinstance(default_val, float):
                kwargs[f.name] = float(val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)

    def add_recent_project(self, path: str, *, max_entries: int = 20) -> None:
        """Add a project path to the front of the recent list (deduped)."""
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        self.recent_projects.insert(0, path)
        self.recent_projects = self.recent_projects[:max_entries]


# ---------------------------------------------------------------------------
# ProjectSettings
# ---------------------------------------------------------------------------


@dataclass
class ProjectSettings:
    """Per-project settings.  Persisted to ``.corridorkey_settings.json``."""

    # Inference
    input_is_linear: bool = False
    despill_strength: float = DEFAULT_DESPILL_STRENGTH
    auto_despeckle: bool = True
    despeckle_size: int = DEFAULT_DESPECKLE_SIZE
    refiner_scale: float = DEFAULT_REFINER_SCALE

    # Alpha generation
    alpha_model: str = "birefnet"
    alpha_mode: str = "replace"

    def save(self, project_root: Path) -> None:
        """Write settings to JSON at *project_root*."""
        dest = project_root / ".corridorkey_settings.json"
        data = asdict(self)
        data["_version"] = _SETTINGS_VERSION
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(dest)

    @classmethod
    def load(cls, project_root: Path) -> ProjectSettings:
        """Load settings from JSON, falling back to defaults for missing keys."""
        src = project_root / ".corridorkey_settings.json"
        if not src.exists():
            return cls()
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse %s, using defaults", src)
            return cls()
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, object]) -> ProjectSettings:
        """Deserialize from a parsed JSON dict, handling missing/extra keys."""
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**kwargs)
