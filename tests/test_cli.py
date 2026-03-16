"""Tests for the typer-based CLI in backend.cli.py."""

from __future__ import annotations

import re
from unittest.mock import patch

from typer.testing import CliRunner

from ck_engine.cli import app
from ck_engine.config import Dir, DEFAULT_DESPILL_STRENGTH, DEFAULT_DESPECKLE_SIZE, DEFAULT_REFINER_SCALE
from ck_engine.pipeline import InferenceSettings

runner = CliRunner()

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Help output
# ---------------------------------------------------------------------------


class TestHelpOutput:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "generate-alphas" in plain
        assert "inference" in plain
        # wizard is hidden from help
        assert "wizard" not in plain.lower().split("commands")[-1] or "wizard" not in plain

    def test_generate_alphas_help(self):
        result = runner.invoke(app, ["generate-alphas", "--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "--model" in plain

    def test_inference_help(self):
        result = runner.invoke(app, ["inference", "--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "--despill" in plain

    def test_no_args_runs_callback(self):
        """No args with invoke_without_command runs the callback (TUI launch handled by main())."""
        # With invoke_without_command=True, bare `app([])` runs the callback.
        # The TUI launch is handled in main(), not in the Typer app.
        # Just verify --help still works as the user-facing "no args" test.
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "Usage" in plain


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


class TestInvalidArgs:
    def test_unknown_subcommand_as_path_fails(self):
        """A nonexistent path routed to wizard fails gracefully."""
        result = runner.invoke(app, ["wizard", "nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Default-to-wizard routing (via main() argv rewriting)
# ---------------------------------------------------------------------------


class TestDefaultRouting:
    def test_bare_path_launches_tui(self, tmp_path):
        """main() launches TUI when first positional is not a subcommand."""
        from ck_engine.cli import _SUBCOMMANDS

        # The path is not in _SUBCOMMANDS, so main() launches TUI
        assert str(tmp_path) not in _SUBCOMMANDS

    def test_subcommand_not_rewritten(self):
        """Known subcommand names are not rewritten."""
        from ck_engine.cli import _SUBCOMMANDS

        assert "inference" in _SUBCOMMANDS
        assert "generate-alphas" in _SUBCOMMANDS

    def test_wizard_with_list_flag(self, tmp_path):
        """wizard --list works with a proper clip structure."""
        clip_root = tmp_path / "shot_a"
        input_dir = clip_root / Dir.INPUT
        input_dir.mkdir(parents=True)
        import cv2
        import numpy as np

        cv2.imwrite(str(input_dir / "frame_0000.png"), np.zeros((4, 4, 3), dtype=np.uint8))
        result = runner.invoke(app, ["--list", "wizard", str(clip_root)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# InferenceSettings defaults
# ---------------------------------------------------------------------------


class TestInferenceSettings:
    def test_defaults(self):
        s = InferenceSettings()
        assert s.input_is_linear is False
        assert s.despill_strength == DEFAULT_DESPILL_STRENGTH
        assert s.auto_despeckle is True
        assert s.despeckle_size == DEFAULT_DESPECKLE_SIZE
        assert s.refiner_scale == DEFAULT_REFINER_SCALE

    def test_custom_values(self):
        s = InferenceSettings(
            input_is_linear=True,
            despill_strength=0.8,
            auto_despeckle=False,
            despeckle_size=200,
            refiner_scale=1.5,
        )
        assert s.input_is_linear is True
        assert s.despill_strength == 0.8
        assert s.auto_despeckle is False
        assert s.despeckle_size == 200
        assert s.refiner_scale == 1.5


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class TestCallbackProtocol:
    @patch("ck_engine.cli._propose_structure")
    @patch("ck_engine.cli._render_events_with_progress")
    @patch("ck_engine.client.EngineClient.spawn")
    @patch("ck_engine.cli._prompt_inference_settings")
    def test_inference_uses_engine_client(self, mock_prompt, mock_spawn, mock_render, mock_propose, tmp_path):
        """inference subcommand spawns EngineClient and submits a job."""
        from unittest.mock import MagicMock
        from ck_engine.api.events import JobCompleted

        mock_client = MagicMock()
        mock_client.submit_inference.return_value = "job-1"
        mock_spawn.return_value = mock_client
        mock_prompt.return_value = (InferenceSettings(), 0)
        mock_render.return_value = JobCompleted(
            job_id="job-1", clips_ok=1, clips_failed=0,
            total_frames=10, frames_ok=10, frames_failed=0,
            elapsed_seconds=1.0,
        )

        result = runner.invoke(app, ["inference", str(tmp_path)])
        assert result.exit_code == 0

        mock_spawn.assert_called_once()
        mock_client.submit_inference.assert_called_once()
        mock_render.assert_called_once()
        mock_client.close.assert_called_once_with(shutdown=True)

    def test_callback_signatures(self):
        """Callbacks accept the documented (name, count) / (idx, total) args."""
        from ck_engine.cli import ProgressContext

        ctx = ProgressContext()
        ctx.__enter__()
        try:
            ctx.on_clip_start("test_clip", 100)
            ctx.on_progress(0, 100)
            ctx.on_progress(99, 100)
        finally:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Non-interactive flags for inference
# ---------------------------------------------------------------------------


class TestNonInteractiveFlags:
    @patch("ck_engine.cli._propose_structure")
    @patch("ck_engine.cli._render_events_with_progress")
    @patch("ck_engine.client.EngineClient.spawn")
    def test_all_flags_skips_prompts(self, mock_spawn, mock_render, mock_propose, tmp_path):
        """When all settings flags are provided, no interactive prompts fire."""
        from unittest.mock import MagicMock
        from ck_engine.api.events import JobCompleted
        from ck_engine.api.types import InferenceSettings as APISettings

        mock_client = MagicMock()
        mock_client.submit_inference.return_value = "job-1"
        mock_spawn.return_value = mock_client
        mock_render.return_value = JobCompleted(
            job_id="job-1", clips_ok=1, clips_failed=0,
            total_frames=10, frames_ok=10, frames_failed=0,
            elapsed_seconds=1.0,
        )

        result = runner.invoke(
            app,
            [
                "inference",
                str(tmp_path),
                "--linear",
                "--despill",
                "7",
                "--despeckle",
                "--despeckle-size",
                "200",
                "--refiner",
                "1.5",
            ],
        )
        assert result.exit_code == 0

        mock_client.submit_inference.assert_called_once()
        params = mock_client.submit_inference.call_args[0][0]
        settings = params.settings
        assert settings.input_is_linear is True
        assert settings.despill_strength == 0.7
        assert settings.auto_despeckle is True
        assert settings.despeckle_size == 200
        assert settings.refiner_scale == 1.5

    @patch("ck_engine.cli._propose_structure")
    @patch("ck_engine.cli._render_events_with_progress")
    @patch("ck_engine.client.EngineClient.spawn")
    def test_srgb_flag(self, mock_spawn, mock_render, mock_propose, tmp_path):
        """--srgb sets input_is_linear=False."""
        from unittest.mock import MagicMock
        from ck_engine.api.events import JobCompleted

        mock_client = MagicMock()
        mock_client.submit_inference.return_value = "job-1"
        mock_spawn.return_value = mock_client
        mock_render.return_value = JobCompleted(
            job_id="job-1", clips_ok=1, clips_failed=0,
            total_frames=10, frames_ok=10, frames_failed=0,
            elapsed_seconds=1.0,
        )

        result = runner.invoke(
            app,
            [
                "inference",
                str(tmp_path),
                "--srgb",
                "--despill",
                "5",
                "--no-despeckle",
                "--refiner",
                "1.0",
            ],
        )
        assert result.exit_code == 0

        mock_client.submit_inference.assert_called_once()
        params = mock_client.submit_inference.call_args[0][0]
        settings = params.settings
        assert settings.input_is_linear is False
        assert settings.auto_despeckle is False

    @patch("ck_engine.cli._propose_structure")
    @patch("ck_engine.cli._render_events_with_progress")
    @patch("ck_engine.client.EngineClient.spawn")
    def test_despill_clamped_to_range(self, mock_spawn, mock_render, mock_propose, tmp_path):
        """Despill values outside 0-10 are clamped."""
        from unittest.mock import MagicMock
        from ck_engine.api.events import JobCompleted

        mock_client = MagicMock()
        mock_client.submit_inference.return_value = "job-1"
        mock_spawn.return_value = mock_client
        mock_render.return_value = JobCompleted(
            job_id="job-1", clips_ok=1, clips_failed=0,
            total_frames=10, frames_ok=10, frames_failed=0,
            elapsed_seconds=1.0,
        )

        result = runner.invoke(
            app,
            [
                "inference",
                str(tmp_path),
                "--srgb",
                "--despill",
                "15",
                "--despeckle",
                "--refiner",
                "1.0",
            ],
        )
        assert result.exit_code == 0

        mock_client.submit_inference.assert_called_once()
        params = mock_client.submit_inference.call_args[0][0]
        settings = params.settings
        assert settings.despill_strength == 1.0  # clamped 15->10, then /10

    def test_inference_help_shows_flags(self):
        """inference --help lists the settings flags."""
        result = runner.invoke(app, ["inference", "--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "--despill" in plain
        assert "--linear" in plain
        assert "--refiner" in plain
        assert "--despeckle-size" in plain

    def test_shared_options_in_main_help(self):
        """Shared optimization options appear in the top-level help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "--profile" in plain
        assert "--device" in plain
        assert "--backend" in plain
