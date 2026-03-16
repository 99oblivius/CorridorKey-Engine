"""Modal editor for inference settings (colorspace, despill, despeckle, refiner)."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch


class InferenceSettingsEditor(ModalScreen[bool]):
    """Modal dialog for editing inference settings."""

    DEFAULT_CSS = """
    InferenceSettingsEditor {
        align: center middle;
    }

    InferenceSettingsEditor #editor-box {
        width: 55;
        height: auto;
        max-height: 24;
        border: solid $border;
        background: $surface;
        padding: 1 2;
    }

    InferenceSettingsEditor #editor-header {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    InferenceSettingsEditor #editor-title {
        width: 1fr;
        content-align: left middle;
        text-style: bold;
        color: $primary;
    }

    InferenceSettingsEditor #ed-save {
        min-width: 16;
    }

    InferenceSettingsEditor .field-row {
        height: 3;
        layout: horizontal;
    }

    InferenceSettingsEditor .field-label {
        width: 16;
        content-align: left middle;
    }

    InferenceSettingsEditor Select {
        width: 1fr;
    }

    InferenceSettingsEditor Input {
        width: 1fr;
    }

    InferenceSettingsEditor Switch {
        width: auto;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Close"),
    ]

    def __init__(
        self,
        settings: object,
        project_path: str | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._project_path = project_path

    def compose(self) -> ComposeResult:
        s = self._settings
        with Vertical(id="editor-box"):
            with Horizontal(id="editor-header"):
                yield Static("Inference Settings", id="editor-title")
                yield Button("Save & Close", variant="primary", id="ed-save")
            with Horizontal(classes="field-row"):
                yield Label("Colorspace:", classes="field-label")
                yield Select(
                    [("sRGB", "srgb"), ("Linear", "linear")],
                    value="linear" if s.input_is_linear else "srgb",
                    allow_blank=False,
                    id="ed-colorspace",
                )
            with Horizontal(classes="field-row"):
                yield Label("Despill (0-10):", classes="field-label")
                yield Input(
                    str(int(s.despill_strength * 10)),
                    type="integer",
                    id="ed-despill",
                )
            with Horizontal(classes="field-row"):
                yield Label("Despeckle:", classes="field-label")
                yield Switch(value=s.auto_despeckle, id="ed-despeckle")
            with Horizontal(classes="field-row"):
                yield Label("Despeckle size:", classes="field-label")
                yield Input(
                    str(s.despeckle_size),
                    type="integer",
                    id="ed-despeckle-size",
                )
            with Horizontal(classes="field-row"):
                yield Label("Refiner scale:", classes="field-label")
                yield Input(
                    str(s.refiner_scale),
                    id="ed-refiner",
                )

    @on(Button.Pressed, "#ed-save")
    def _on_save(self, event: Button.Pressed) -> None:
        self._save_and_close()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _save_and_close(self) -> None:
        """Read values from widgets, save to project, dismiss."""
        colorspace = self.query_one("#ed-colorspace", Select)
        despill = self.query_one("#ed-despill", Input)
        despeckle = self.query_one("#ed-despeckle", Switch)
        despeckle_size = self.query_one("#ed-despeckle-size", Input)
        refiner = self.query_one("#ed-refiner", Input)

        try:
            despill_val = max(0, min(10, int(despill.value))) / 10.0
        except ValueError:
            despill_val = 0.5
        try:
            despeckle_size_val = max(0, int(despeckle_size.value))
        except ValueError:
            despeckle_size_val = 400
        try:
            refiner_val = max(0.0, float(refiner.value))
        except ValueError:
            refiner_val = 1.0

        if self._project_path:
            try:
                from pathlib import Path

                from ck_engine.settings import ProjectSettings

                ps = ProjectSettings(
                    input_is_linear=colorspace.value == "linear",
                    despill_strength=despill_val,
                    auto_despeckle=despeckle.value,
                    despeckle_size=despeckle_size_val,
                    refiner_scale=refiner_val,
                )
                ps.save(Path(self._project_path))
            except Exception:
                pass

        self.dismiss(True)
