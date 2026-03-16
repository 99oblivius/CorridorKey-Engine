"""Global Settings panel — optimization flags, device config, pipeline tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static, Switch

from ..settings_io import GlobalSettings

if TYPE_CHECKING:
    from textual.app import ComposeResult


# Optimization profile presets: profile_name -> {field: value}
_PROFILES: dict[str, dict[str, bool | None]] = {
    "optimized": {
        "flash_attention": True,
        "tiled_refiner": True,
        "cache_clearing": True,
        "disable_cudnn_benchmark": None,
        "gpu_postprocess": True,
    },
    "original": {
        "flash_attention": False,
        "tiled_refiner": False,
        "cache_clearing": False,
        "disable_cudnn_benchmark": None,
        "gpu_postprocess": None,
    },
    "performance": {
        "flash_attention": True,
        "tiled_refiner": True,
        "cache_clearing": False,
        "disable_cudnn_benchmark": False,
        "gpu_postprocess": True,
    },
}


def _section(title: str) -> Static:
    """Create a section header label."""
    return Static(f"── {title} ──", classes="section-header")


def _field_row(label: str, widget_id: str, widget: object) -> Horizontal:
    """Create a label + widget row."""
    return Horizontal(
        Label(label, classes="field-label"),
        widget,
        classes="field-row",
        id=f"row-{widget_id}",
    )


class GlobalSettingsPanel(Vertical, can_focus=True):
    """Global settings for device, pipeline, and optimization configuration."""

    DEFAULT_CSS = """
    GlobalSettingsPanel {
        layout: vertical;
    }

    GlobalSettingsPanel #settings-title {
        height: 1;
        text-style: bold;
        color: $primary;
        padding: 0 1;
        margin-bottom: 1;
    }

    GlobalSettingsPanel #settings-scroll {
        height: 1fr;
    }

    GlobalSettingsPanel .section-header {
        height: 1;
        text-style: bold;
        color: $secondary;
        padding: 0 1;
        margin-top: 1;
    }

    GlobalSettingsPanel .field-row {
        height: 3;
        padding: 0 1;
    }

    GlobalSettingsPanel .field-label {
        width: 22;
        content-align: left middle;
    }

    GlobalSettingsPanel .field-row Input {
        width: 20;
    }

    GlobalSettingsPanel .field-row Select {
        width: 30;
    }

    GlobalSettingsPanel .field-row Switch {
        width: 12;
    }

    GlobalSettingsPanel #button-bar {
        dock: bottom;
        height: auto;
        min-height: 3;
        padding: 1;
        layout: horizontal;
    }

    GlobalSettingsPanel #button-bar Button {
        margin-right: 2;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+s", "save_settings", "Save", priority=True),
        Binding("escape", "blur_field", "Back", show=False, priority=True),
    ]

    _OPT_SWITCH_IDS: ClassVar[set[str]] = {
        "flash_attention", "tiled_refiner", "cache_clearing",
        "cudnn_benchmark", "gpu_postprocess",
    }

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._settings = GlobalSettings.load()
        self._ignore_switch_count = 0  # Skip N switch-changed events (from profile/reset)

    def compose(self) -> ComposeResult:
        s = self._settings
        yield Static("GLOBAL SETTINGS", id="settings-title")
        with VerticalScroll(id="settings-scroll"):
            # -- Device --
            yield _section("Device")
            yield _field_row(
                "Device:",
                "device",
                Select(
                    [("auto", "auto"), ("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")],
                    value=s.device,
                    allow_blank=False,
                    id="device",
                ),
            )
            yield _field_row(
                "Backend:",
                "backend",
                Select(
                    [("auto", "auto"), ("torch", "torch"), ("torch_optimized", "torch_optimized"), ("mlx", "mlx")],
                    value=s.backend,
                    allow_blank=False,
                    id="backend",
                ),
            )
            yield _field_row(
                "Precision:",
                "precision",
                Select(
                    [("fp16", "fp16"), ("bf16", "bf16"), ("fp32", "fp32")],
                    value=s.precision,
                    allow_blank=False,
                    id="precision",
                ),
            )
            yield _field_row(
                "GPU Indices:",
                "devices",
                Input(
                    value=",".join(s.devices) if s.devices else "",
                    placeholder="e.g. 0,1 (empty=all)",
                    id="devices",
                ),
            )

            # -- Pipeline --
            yield _section("Pipeline")
            yield _field_row(
                "Image Size:",
                "img_size",
                Input(
                    value=str(s.img_size),
                    id="img_size",
                ),
            )
            yield _field_row(
                "CPU Workers:",
                "cpus",
                Input(
                    value=str(s.cpus),
                    placeholder="0 = auto",
                    id="cpus",
                ),
            )
            yield _field_row(
                "Read Workers:",
                "read_workers",
                Input(
                    value=str(s.read_workers),
                    placeholder="0 = auto",
                    id="read_workers",
                ),
            )
            yield _field_row(
                "Write Workers:",
                "write_workers",
                Input(
                    value=str(s.write_workers),
                    placeholder="0 = auto",
                    id="write_workers",
                ),
            )
            yield _field_row(
                "GPU Resilience:",
                "gpu_resilience",
                Switch(
                    value=s.gpu_resilience,
                    id="gpu_resilience",
                ),
            )

            # -- Optimization --
            yield _section("Optimization")
            yield _field_row(
                "Profile:",
                "profile",
                Select(
                    [
                        ("(custom)", ""),
                        ("optimized", "optimized"),
                        ("original", "original"),
                        ("performance", "performance"),
                    ],
                    value=s.profile or "",
                    allow_blank=False,
                    id="profile",
                ),
            )
            yield _field_row(
                "Flash Attention:",
                "flash_attention",
                Switch(
                    value=s.flash_attention is True,
                    id="flash_attention",
                ),
            )
            yield _field_row(
                "Tiled Refiner:",
                "tiled_refiner",
                Switch(
                    value=s.tiled_refiner is True,
                    id="tiled_refiner",
                ),
            )
            yield _field_row(
                "Tile Size:",
                "tile_size",
                Input(
                    value=str(s.tile_size),
                    id="tile_size",
                ),
            )
            yield _field_row(
                "Tile Overlap:",
                "tile_overlap",
                Input(
                    value=str(s.tile_overlap),
                    id="tile_overlap",
                ),
            )
            yield _field_row(
                "Cache Clearing:",
                "cache_clearing",
                Switch(
                    value=s.cache_clearing is True,
                    id="cache_clearing",
                ),
            )
            yield _field_row(
                "cuDNN Benchmark:",
                "cudnn_benchmark",
                Switch(
                    value=s.disable_cudnn_benchmark is not True,
                    id="cudnn_benchmark",
                ),
            )
            yield _field_row(
                "Compile Mode:",
                "compile_mode",
                Select(
                    [
                        ("(none)", "none"),
                        ("default", "default"),
                        ("reduce-overhead", "reduce-overhead"),
                        ("max-autotune", "max-autotune"),
                    ],
                    value=s.compile_mode or "none",
                    allow_blank=False,
                    id="compile_mode",
                ),
            )
            yield _field_row(
                "GPU Postprocess:",
                "gpu_postprocess",
                Switch(
                    value=s.gpu_postprocess is not False,
                    id="gpu_postprocess",
                ),
            )

            # -- Output --
            yield _section("Output")
            yield _field_row(
                "Comp Format:",
                "comp_format",
                Select(
                    [("exr", "exr"), ("png", "png"), ("none", "none")],
                    value=s.comp_format,
                    allow_blank=False,
                    id="comp_format",
                ),
            )
            yield _field_row(
                "Checkerboard:",
                "comp_checkerboard",
                Switch(
                    value=s.comp_checkerboard,
                    id="comp_checkerboard",
                ),
            )
            yield _field_row(
                "DMA Buffers:",
                "dma_buffers",
                Input(
                    value=str(s.dma_buffers),
                    id="dma_buffers",
                ),
            )

        with Horizontal(id="button-bar"):
            yield Button("Save", variant="primary", id="save-btn")
            yield Button("Reset to Defaults", variant="default", id="reset-btn")

    @on(Select.Changed, "#profile")
    def _on_profile_changed(self, event: Select.Changed) -> None:
        """Auto-fill optimization toggles when a profile is selected."""
        profile_name = str(event.value) if event.value != Select.BLANK else ""
        if profile_name not in _PROFILES:
            return
        preset = _PROFILES[profile_name]
        changed = 0
        for field_name, value in preset.items():
            widget_id = field_name
            if field_name == "disable_cudnn_benchmark":
                widget_id = "cudnn_benchmark"
                value = value is not True  # Invert for display
            try:
                switch = self.query_one(f"#{widget_id}", Switch)
                old = switch.value
                switch.value = bool(value)
                if old != bool(value):
                    changed += 1
            except Exception:
                pass
        self._ignore_switch_count += changed

    @on(Switch.Changed)
    def _on_switch_changed(self, event: Switch.Changed) -> None:
        """Reset profile to (custom) when user manually toggles an opt switch."""
        widget_id = event.switch.id or ""
        if widget_id not in self._OPT_SWITCH_IDS:
            return
        if self._ignore_switch_count > 0:
            self._ignore_switch_count -= 1
            return
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one("#profile", Select).value = ""

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        self._collect_and_save()

    @on(Button.Pressed, "#reset-btn")
    def _on_reset(self, event: Button.Pressed) -> None:
        """Reset all widget values to defaults in-place (doesn't save until Save)."""
        d = GlobalSettings()
        self._settings = d
        # Count opt switches that will change to suppress profile→custom cascade
        opt_switch_count = len(self._OPT_SWITCH_IDS)
        self._ignore_switch_count += opt_switch_count

        # Selects
        self._set_select("device", d.device)
        self._set_select("backend", d.backend)
        self._set_select("precision", d.precision)
        self._set_select("profile", d.profile or "")
        self._set_select("compile_mode", d.compile_mode or "none")
        self._set_select("comp_format", d.comp_format)

        # Inputs
        self._set_input("devices", "")
        self._set_input("img_size", str(d.img_size))
        self._set_input("cpus", str(d.cpus))
        self._set_input("read_workers", str(d.read_workers))
        self._set_input("write_workers", str(d.write_workers))
        self._set_input("tile_size", str(d.tile_size))
        self._set_input("tile_overlap", str(d.tile_overlap))
        self._set_input("dma_buffers", str(d.dma_buffers))

        # Switches
        self._set_switch("gpu_resilience", d.gpu_resilience)
        self._set_switch("flash_attention", d.flash_attention is True)
        self._set_switch("tiled_refiner", d.tiled_refiner is True)
        self._set_switch("cache_clearing", d.cache_clearing is True)
        self._set_switch("cudnn_benchmark", d.disable_cudnn_benchmark is not True)
        self._set_switch("gpu_postprocess", d.gpu_postprocess is not False)
        self._set_switch("comp_checkerboard", d.comp_checkerboard)

        self.notify("Reset to defaults. Press Save to persist.", severity="information")

    def action_blur_field(self) -> None:
        """Move focus out of the current field back to the scroll area."""
        scroll = self.query_one("#settings-scroll", VerticalScroll)
        scroll.focus()

    def action_save_settings(self) -> None:
        self._collect_and_save()

    def _collect_and_save(self) -> None:
        """Read all widget values, update settings, save to TOML."""
        s = self._settings

        # Device
        s.device = self._select_val("device", s.device)
        s.backend = self._select_val("backend", s.backend)
        s.precision = self._select_val("precision", s.precision)
        devices_text = self._input_val("devices", "")
        s.devices = [d.strip() for d in devices_text.split(",") if d.strip()] if devices_text else []

        # Pipeline
        s.img_size = self._input_int("img_size", s.img_size)
        s.cpus = self._input_int("cpus", s.cpus)
        s.read_workers = self._input_int("read_workers", s.read_workers)
        s.write_workers = self._input_int("write_workers", s.write_workers)
        s.gpu_resilience = self._switch_val("gpu_resilience")

        # Optimization
        s.profile = self._select_val("profile", s.profile)
        s.flash_attention = self._switch_val("flash_attention")
        s.tiled_refiner = self._switch_val("tiled_refiner")
        s.tile_size = self._input_int("tile_size", s.tile_size)
        s.tile_overlap = self._input_int("tile_overlap", s.tile_overlap)
        s.cache_clearing = self._switch_val("cache_clearing")
        s.disable_cudnn_benchmark = not self._switch_val("cudnn_benchmark")
        s.compile_mode = self._select_val("compile_mode", s.compile_mode)
        s.gpu_postprocess = self._switch_val("gpu_postprocess")

        # Output
        s.comp_format = self._select_val("comp_format", s.comp_format)
        s.comp_checkerboard = self._switch_val("comp_checkerboard")
        s.dma_buffers = self._input_int("dma_buffers", s.dma_buffers)

        s.save()
        self.notify("Settings saved.", severity="information")

    def _select_val(self, widget_id: str, default: str) -> str:
        try:
            sel = self.query_one(f"#{widget_id}", Select)
            return str(sel.value) if sel.value != Select.BLANK else default
        except Exception:
            return default

    def _input_val(self, widget_id: str, default: str) -> str:
        try:
            return self.query_one(f"#{widget_id}", Input).value
        except Exception:
            return default

    def _input_int(self, widget_id: str, default: int) -> int:
        try:
            return int(self.query_one(f"#{widget_id}", Input).value)
        except (ValueError, Exception):
            return default

    def _switch_val(self, widget_id: str) -> bool:
        try:
            return self.query_one(f"#{widget_id}", Switch).value
        except Exception:
            return False

    # -- Setters for reset --

    def _set_select(self, widget_id: str, value: str) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one(f"#{widget_id}", Select).value = value

    def _set_input(self, widget_id: str, value: str) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one(f"#{widget_id}", Input).value = value

    def _set_switch(self, widget_id: str, value: bool) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one(f"#{widget_id}", Switch).value = value
