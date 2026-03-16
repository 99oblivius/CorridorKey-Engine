"""CorridorKey Engine — command-line interface and interactive wizard.

Usage:
    corridorkey-engine                                    # launch TUI
    corridorkey-engine /path/to/clips [OPTIONS]           # TUI with path
    corridorkey-engine inference /path [OPTIONS]           # headless inference
    corridorkey-engine generate-alphas /path --model gvm   # alpha generation
    corridorkey-engine --help                              # global help
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import re
import sys
import warnings
from typing import TYPE_CHECKING, Annotated, Optional, Self

if TYPE_CHECKING:
    from .pipeline import AlphaMode, InferenceSettings

import typer
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .api.errors import EngineError
from .config import DEFAULT_IMG_SIZE, DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP, Dir
from .path_utils import LINUX_MOUNT_ROOT, map_path

logger = logging.getLogger(__name__)
console = Console()


# Known subcommand names — used by main() to detect wizard-mode invocations.
_SUBCOMMANDS = {"generate-alphas", "inference", "serve"}


app = typer.Typer(
    name="corridorkey-engine",
    help="Neural network green screen keying for professional VFX pipelines.",
    rich_markup_mode="rich",
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": False},
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=False, highlighter=NullHighlighter())],
    )


# ---------------------------------------------------------------------------
# Readline-safe input helper
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _readline_input(markup: str, *, suffix: str = ": ") -> str:
    """Prompt with Rich markup, safe for readline/backspace.

    Renders *markup* to an ANSI string, wraps escape sequences in
    readline ignore markers (``\\x01``/``\\x02``), then passes the
    result to :func:`input`.  This lets readline track cursor position
    correctly so backspace never erases the prompt text.
    """
    with console.capture() as cap:
        console.print(markup, end="")
    ansi = cap.get() + suffix
    safe = _ANSI_RE.sub(lambda m: "\x01" + m.group(1) + "\x02", ansi)
    return input(safe)


# ---------------------------------------------------------------------------
# Progress helpers (callback protocol -> rich.progress)
# ---------------------------------------------------------------------------


class _FrameSpeedColumn(ProgressColumn):
    """Render frame throughput as ``X.XX frame/s``."""

    def render(self, task: Task) -> str:
        speed = task.finished_speed or task.speed
        if speed is None:
            return ""
        return f"{speed:.2f} frame/s"


def _fmt_bytes_speed(bps: float) -> str:
    """Format bytes/second as human-readable throughput."""
    if bps < 1024:
        return f"{bps:.0f} B/s"
    if bps < 1024 * 1024:
        return f"{bps / 1024:.1f} KB/s"
    if bps < 1024 * 1024 * 1024:
        return f"{bps / (1024 * 1024):.1f} MB/s"
    return f"{bps / (1024 * 1024 * 1024):.2f} GB/s"


class _IOSpeedTracker:
    """3-second exponentially-weighted moving average for IO throughput.

    Receives cumulative byte totals at irregular intervals. Computes
    instantaneous rates between consecutive samples and decays them
    exponentially — recent activity dominates, stale samples fade.
    Samples older than 6s (2x window) are pruned.
    """

    def __init__(self, window: float = 3.0) -> None:
        self._window = window
        self._samples_r: list[tuple[float, int]] = []
        self._samples_w: list[tuple[float, int]] = []

    def record(self, now: float, total_read: int, total_written: int) -> None:
        self._samples_r.append((now, total_read))
        self._samples_w.append((now, total_written))
        cutoff = now - self._window * 2
        self._samples_r = [(t, b) for t, b in self._samples_r if t >= cutoff]
        self._samples_w = [(t, b) for t, b in self._samples_w if t >= cutoff]

    def speeds(self) -> tuple[float, float]:
        return (self._ewma(self._samples_r), self._ewma(self._samples_w))

    def _ewma(self, samples: list[tuple[float, int]]) -> float:
        if len(samples) < 2:
            return 0.0
        now = samples[-1][0]
        tau = self._window
        weighted_rate = 0.0
        total_weight = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i - 1][0]
            if dt <= 0:
                continue
            rate = (samples[i][1] - samples[i - 1][1]) / dt
            age = now - samples[i][0]
            weight = 2.0 ** (-age / tau)
            weighted_rate += rate * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_rate / total_weight


class _IOSpeedColumn(ProgressColumn):
    """Render file IO read/write speeds from pre-computed task fields."""

    def render(self, task: Task) -> str:
        r_speed = task.fields.get("io_r_speed", 0.0)
        w_speed = task.fields.get("io_w_speed", 0.0)
        if r_speed == 0.0 and w_speed == 0.0:
            return ""
        return f"R {_fmt_bytes_speed(r_speed)} | W {_fmt_bytes_speed(w_speed)}"


class ProgressContext:
    """Context manager bridging clip_manager callbacks to Rich progress bars.

    The async pipeline reports *absolute* completed counts via
    ``on_progress(completed, total, bytes_read, bytes_written)``
    from a background thread. IO speeds are computed as a 3-second
    exponentially-decaying average from cumulative byte totals.
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            _FrameSpeedColumn(),
            _IOSpeedColumn(),
            console=console,
            expand=True,
        )
        self._frame_task_id: TaskID | None = None
        self._io_tracker = _IOSpeedTracker(window=3.0)

    def __enter__(self) -> Self:
        self._progress.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.__exit__(*exc)

    def on_clip_start(self, clip_name: str, num_frames: int) -> None:
        """Callback: reset the progress bar for a new clip."""
        if self._frame_task_id is not None:
            self._progress.remove_task(self._frame_task_id)
        self._frame_task_id = self._progress.add_task(f"[cyan]{clip_name}", total=num_frames)
        self._io_tracker = _IOSpeedTracker(window=3.0)

    def on_progress(
        self,
        completed: int,
        num_frames: int,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> None:
        """Callback: set progress to the absolute completed count."""
        if self._frame_task_id is not None:
            import time as _time

            self._io_tracker.record(_time.monotonic(), bytes_read, bytes_written)
            r_speed, w_speed = self._io_tracker.speeds()
            self._progress.update(
                self._frame_task_id,
                completed=completed,
                io_r_speed=r_speed,
                io_w_speed=w_speed,
            )


def _on_clip_start_log_only(clip_name: str, total_clips: int) -> None:
    """Clip-level callback for generate-alphas."""
    console.print(f"  Processing [bold]{clip_name}[/bold] ({total_clips} total)")


# ---------------------------------------------------------------------------
# Inference settings prompt (readline-safe -- CLI layer only)
# ---------------------------------------------------------------------------


def _prompt_inference_settings(
    *,
    default_linear: bool | None = None,
    default_despill: int | None = None,
    default_despeckle: bool | None = None,
    default_despeckle_size: int | None = None,
    default_refiner: float | None = None,
) -> tuple[InferenceSettings | None, int]:
    """Interactively prompt for inference settings, skipping any pre-filled values.

    Returns (settings, lines_printed). *lines_printed* counts terminal lines
    produced so the caller can erase them on cancel.
    """
    from .pipeline import InferenceSettings

    _defaults = InferenceSettings()
    lines = 0
    with console.capture() as cap_hdr:
        console.print(Panel("Inference Settings", style="bold cyan"))
    hdr = cap_hdr.get()
    lines += hdr.count("\n")
    sys.stdout.write(hdr)
    sys.stdout.flush()

    try:
        if default_linear is not None:
            input_is_linear = default_linear
        else:
            while True:
                _linear_hint = "linear" if _defaults.input_is_linear else "srgb"
                gamma_choice = _readline_input(
                    "Input colorspace"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]l[/bold magenta][magenta]inear[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]s[/bold magenta][magenta]rgb[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    f" [cyan]({_linear_hint})[/cyan]",
                )
                val = gamma_choice.strip().lower()
                if not val:
                    input_is_linear = _defaults.input_is_linear
                    lines += 1
                    break
                if val in ("l", "linear"):
                    input_is_linear = True
                    lines += 1
                    break
                if val in ("s", "srgb"):
                    input_is_linear = False
                    lines += 1
                    break

        if default_despill is not None:
            despill_int = max(0, min(10, default_despill))
        else:
            while True:
                _despill_default = round(_defaults.despill_strength * 10)
                raw = _readline_input(
                    f"Despill strength [cyan](0-10, 10 = max despill)[/cyan] [cyan]({_despill_default})[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despill_int = _despill_default
                    lines += 1
                    break
                try:
                    despill_int = int(val)
                    despill_int = max(0, min(10, despill_int))
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()
        despill_strength = despill_int / 10.0

        if default_despeckle is not None:
            auto_despeckle = default_despeckle
        else:
            while True:
                raw = _readline_input(
                    "Enable auto-despeckle (removes tracking dots)?"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    f" [cyan]({'yes' if _defaults.auto_despeckle else 'no'})[/cyan]",
                )
                val = raw.strip().lower()
                if not val:
                    auto_despeckle = _defaults.auto_despeckle
                    lines += 1
                    break
                if val in ("n", "no"):
                    auto_despeckle = False
                    lines += 1
                    break
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

        despeckle_size = default_despeckle_size if default_despeckle_size is not None else _defaults.despeckle_size
        if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
            while True:
                raw = _readline_input(
                    f"Despeckle size [cyan](min pixels for a spot)[/cyan] [cyan]({_defaults.despeckle_size})[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despeckle_size = _defaults.despeckle_size
                    lines += 1
                    break
                try:
                    despeckle_size = max(0, int(val))
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()

        if default_refiner is not None:
            refiner_scale = default_refiner
        else:
            while True:
                raw = _readline_input(
                    f"Refiner strength multiplier [dim](experimental)[/dim] [cyan]({_defaults.refiner_scale})[/cyan]",
                )
                val = raw.strip()
                if not val:
                    refiner_scale = _defaults.refiner_scale
                    lines += 1
                    break
                try:
                    refiner_scale = float(val)
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()

        return InferenceSettings(
            input_is_linear=input_is_linear,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            refiner_scale=refiner_scale,
        ), lines
    except EOFError:
        return None, lines


# ---------------------------------------------------------------------------
# Optimization config builder
# ---------------------------------------------------------------------------


def _build_optimization_config(ctx_obj: dict) -> object | None:
    """Build an OptimizationConfig from parsed callback params in ctx.obj.

    Returns None if no profile or optimization flags were specified.
    """
    from dataclasses import replace

    from CorridorKeyModule.optimization_config import OptimizationConfig

    backend = ctx_obj["backend"]
    profile = ctx_obj["profile"]
    flash_attention = ctx_obj["flash_attention"]
    tiled_refiner = ctx_obj["tiled_refiner"]
    cache_clearing = ctx_obj["cache_clearing"]
    disable_cudnn_benchmark = ctx_obj["disable_cudnn_benchmark"]
    token_routing = ctx_obj["token_routing"]
    compile_mode = ctx_obj["compile_mode"]
    tensorrt = ctx_obj["tensorrt"]
    tile_size = ctx_obj["tile_size"]
    tile_overlap = ctx_obj["tile_overlap"]
    gpu_postprocess = ctx_obj["gpu_postprocess"]
    comp_format = ctx_obj["comp_format"]
    comp_checkerboard = ctx_obj["comp_checkerboard"]
    dma_buffers = ctx_obj["dma_buffers"]
    precision = ctx_obj["precision"]

    if profile:
        config = OptimizationConfig.from_profile(profile)
    elif backend == "torch_optimized":
        config = OptimizationConfig.optimized()
    elif backend == "torch":
        config = OptimizationConfig.original()
    else:
        has_overrides = (
            any(
                v is not None
                for v in [
                    flash_attention,
                    tiled_refiner,
                    cache_clearing,
                    disable_cudnn_benchmark,
                    compile_mode,
                    gpu_postprocess,
                ]
            )
            or comp_checkerboard
            or comp_format != "exr"
            or token_routing
            or tensorrt
            or dma_buffers != 2
            or precision != "fp16"
            or tile_size != DEFAULT_TILE_SIZE
            or tile_overlap != DEFAULT_TILE_OVERLAP
        )
        if not has_overrides:
            return None
        config = OptimizationConfig()

    overrides = {}
    if flash_attention is not None:
        overrides["flash_attention"] = flash_attention
    if tiled_refiner is not None:
        overrides["tiled_refiner"] = tiled_refiner
    if cache_clearing is not None:
        overrides["cache_clearing"] = cache_clearing
    if disable_cudnn_benchmark is not None:
        overrides["disable_cudnn_benchmark"] = not disable_cudnn_benchmark
    if token_routing:
        overrides["token_routing"] = True
    if compile_mode is not None:
        overrides["compile_mode"] = compile_mode
    if tensorrt:
        overrides["tensorrt"] = True
    if tile_size != DEFAULT_TILE_SIZE:
        overrides["tile_size"] = tile_size
    if tile_overlap != DEFAULT_TILE_OVERLAP:
        overrides["tile_overlap"] = tile_overlap
    if gpu_postprocess is not None:
        overrides["gpu_postprocess"] = gpu_postprocess
    if comp_format != "exr":
        overrides["comp_format"] = comp_format
    if comp_checkerboard:
        overrides["comp_checkerboard"] = True
    if dma_buffers != 2:
        overrides["dma_buffers"] = dma_buffers
    _precision_aliases = {
        "fp16": "float16",
        "fp32": "float32",
        "bf16": "bfloat16",
        "half": "float16",
        "float": "float32",
    }
    resolved_precision = _precision_aliases.get(precision, precision)
    overrides["model_precision"] = resolved_precision

    if overrides:
        config = replace(config, **overrides)

    return config


def _build_optimization_params(ctx_obj: dict) -> OptimizationParams | None:
    """Build API OptimizationParams from CLI arguments."""
    from .api.types import OptimizationParams

    # Check if any optimization flags were explicitly set
    has_opt = any(ctx_obj.get(k) is not None for k in (
        "profile", "flash_attention", "tiled_refiner", "cache_clearing",
        "disable_cudnn_benchmark", "compile_mode", "gpu_postprocess",
    )) or ctx_obj.get("token_routing") or ctx_obj.get("tensorrt")

    if not has_opt:
        return None

    return OptimizationParams(
        profile=ctx_obj.get("profile"),
        flash_attention=ctx_obj.get("flash_attention"),
        tiled_refiner=ctx_obj.get("tiled_refiner"),
        cache_clearing=ctx_obj.get("cache_clearing"),
        disable_cudnn_benchmark=ctx_obj.get("disable_cudnn_benchmark"),
        compile_mode=ctx_obj.get("compile_mode"),
        model_precision=ctx_obj.get("precision"),
        gpu_postprocess=ctx_obj.get("gpu_postprocess"),
        comp_format=ctx_obj.get("comp_format"),
        comp_checkerboard=ctx_obj.get("comp_checkerboard"),
        dma_buffers=ctx_obj.get("dma_buffers"),
        token_routing=ctx_obj.get("token_routing") or None,
        tensorrt=ctx_obj.get("tensorrt") or None,
        tile_size=ctx_obj.get("tile_size"),
        tile_overlap=ctx_obj.get("tile_overlap"),
    )


# ---------------------------------------------------------------------------
# Event-driven progress renderer (EngineClient)
# ---------------------------------------------------------------------------


def _render_events_with_progress(client, job_id):
    """Consume events from the client and render them as Rich progress bars."""
    import queue as _queue
    from .api.events import (
        ClipStarted, JobProgress, ClipCompleted,
        JobCompleted, JobFailed, JobCancelled,
        ModelLoading, ModelLoaded, LogEvent,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        _FrameSpeedColumn(),
        _IOSpeedColumn(),
        console=console,
        expand=True,
    ) as progress:
        frame_task = None
        io_tracker = _IOSpeedTracker(window=3.0)

        try:
            for event in client.iter_events(timeout=300.0):
                if isinstance(event, ClipStarted):
                    if frame_task is not None:
                        progress.remove_task(frame_task)
                    frame_task = progress.add_task(f"[cyan]{event.clip}", total=event.frames)
                    io_tracker = _IOSpeedTracker(window=3.0)

                elif isinstance(event, JobProgress):
                    if frame_task is not None:
                        import time as _time
                        io_tracker.record(_time.monotonic(), event.bytes_read, event.bytes_written)
                        r_speed, w_speed = io_tracker.speeds()
                        progress.update(
                            frame_task,
                            completed=event.done,
                            io_r_speed=r_speed,
                            io_w_speed=w_speed,
                        )

                elif isinstance(event, ModelLoading):
                    console.print(f"  Loading model [bold]{event.model}[/bold] on {event.device}...")

                elif isinstance(event, ModelLoaded):
                    console.print(f"  Model loaded in {event.load_seconds:.1f}s ({event.vram_mb:.0f} MB VRAM)")

                elif isinstance(event, LogEvent):
                    # Show log messages above the progress bar
                    if event.level in ("warning", "error"):
                        color = "yellow" if event.level == "warning" else "red"
                        console.print(f"  [{color}]{event.message}[/{color}]")
                    else:
                        console.print(f"  {event.message}")

                elif isinstance(event, JobCompleted):
                    return event  # success

                elif isinstance(event, JobFailed):
                    return event  # failure

                elif isinstance(event, JobCancelled):
                    return event  # cancelled

        except _queue.Empty:
            console.print("[red]Timed out waiting for engine response.[/red]")
            return None

    return None


# ---------------------------------------------------------------------------
# Typer callback — shared options for all subcommands
# ---------------------------------------------------------------------------


_PANEL_DEVICE = "Device & Pipeline"
_PANEL_OPT = "Optimization"
_PANEL_OUTPUT = "Output"


@app.callback()
def app_callback(
    ctx: typer.Context,
    # --- Device & Pipeline ---
    device: Annotated[
        str,
        typer.Option(help="Compute device: auto, cuda, mps, cpu", rich_help_panel=_PANEL_DEVICE),
    ] = "auto",
    backend: Annotated[
        str,
        typer.Option(help="Backend: auto, torch, torch_optimized, mlx", rich_help_panel=_PANEL_DEVICE),
    ] = "auto",
    devices: Annotated[
        Optional[str],
        typer.Option("--devices", help="Comma-separated GPU indices (e.g. 0,1)", rich_help_panel=_PANEL_DEVICE),
    ] = None,
    img_size: Annotated[
        int,
        typer.Option("--img-size", help="Model input resolution", rich_help_panel=_PANEL_DEVICE),
    ] = DEFAULT_IMG_SIZE,
    read_workers: Annotated[
        int,
        typer.Option("--read-workers", help="Reader thread pool size (0=auto)", rich_help_panel=_PANEL_DEVICE),
    ] = 0,
    write_workers: Annotated[
        int,
        typer.Option("--write-workers", help="Writer thread pool size (0=auto)", rich_help_panel=_PANEL_DEVICE),
    ] = 0,
    cpus: Annotated[
        int,
        typer.Option("--cpus", help="CPU count for worker scaling (0=all available)", rich_help_panel=_PANEL_DEVICE),
    ] = 0,
    gpu_resilience: Annotated[
        bool,
        typer.Option("--gpu-resilience", help="Continue on remaining GPUs if one hits OOM (farm mode)", rich_help_panel=_PANEL_DEVICE),
    ] = False,
    # --- Optimization ---
    profile: Annotated[
        Optional[str],
        typer.Option(
            "--profile",
            help="Optimization profile: optimized, original, experimental*, performance*",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    flash_attention: Annotated[
        Optional[bool],
        typer.Option(
            "--flash-attention/--no-flash-attention",
            help="FlashAttention patching (optimized: on)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    tiled_refiner: Annotated[
        Optional[bool],
        typer.Option(
            "--tiled-refiner/--no-tiled-refiner",
            help="Tiled CNN refiner (optimized: on)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    cache_clearing: Annotated[
        Optional[bool],
        typer.Option(
            "--cache-clearing/--no-cache-clearing",
            help="CUDA cache clearing (optimized: on)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    disable_cudnn_benchmark: Annotated[
        Optional[bool],
        typer.Option(
            "--cudnn-benchmark/--no-cudnn-benchmark",
            help="cuDNN auto-tune, faster, +2-5 GB VRAM",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    token_routing: Annotated[
        bool,
        typer.Option(
            "--token-routing",
            help="Experimental token routing (improves speed at high res)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = False,
    compile_mode: Annotated[
        Optional[str],
        typer.Option(
            "--compile-mode",
            help="torch.compile mode: default, reduce-overhead*, max-autotune*",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    tensorrt: Annotated[
        bool,
        typer.Option(
            "--tensorrt",
            help="TensorRT compilation (broken, WIP)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = False,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="Tile size for tiled refiner", rich_help_panel=_PANEL_OPT),
    ] = DEFAULT_TILE_SIZE,
    tile_overlap: Annotated[
        int,
        typer.Option("--tile-overlap", help="Tile overlap in pixels", rich_help_panel=_PANEL_OPT),
    ] = DEFAULT_TILE_OVERLAP,
    gpu_postprocess: Annotated[
        Optional[bool],
        typer.Option(
            "--gpu-postprocess/--cpu-postprocess",
            help="Postprocessing device (GPU default, +~1.5 GB VRAM)",
            rich_help_panel=_PANEL_OPT,
        ),
    ] = None,
    precision: Annotated[
        str,
        typer.Option("--precision", help="Model weight precision: fp16, bf16, fp32", rich_help_panel=_PANEL_OPT),
    ] = "fp16",
    dma_buffers: Annotated[
        int,
        typer.Option("--dma-buffers", help="Pinned DMA buffer count 2-3", rich_help_panel=_PANEL_OPT),
    ] = 2,
    # --- Output ---
    comp_format: Annotated[
        str,
        typer.Option(
            "--comp",
            help="Composite output format: exr, png, none",
            rich_help_panel=_PANEL_OUTPUT,
        ),
    ] = "exr",
    comp_checkerboard: Annotated[
        bool,
        typer.Option(
            "--checkerboard",
            help="Checkerboard background for composite (not transparent)",
            rich_help_panel=_PANEL_OUTPUT,
        ),
    ] = False,
    # --- Wizard ---
    list_only: Annotated[
        bool,
        typer.Option("--list", help="List clips and exit (wizard mode)"),
    ] = False,
) -> None:
    """Neural network green screen keying for professional VFX pipelines."""
    from .device import resolve_device

    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
    ctx.obj["backend"] = backend
    ctx.obj["devices_list"] = [f"cuda:{idx.strip()}" for idx in devices.split(",")] if devices else None
    ctx.obj["img_size"] = img_size
    ctx.obj["read_workers"] = read_workers
    ctx.obj["write_workers"] = write_workers
    ctx.obj["cpus"] = cpus
    ctx.obj["gpu_resilience"] = gpu_resilience
    # Store raw optimization values for deferred config build
    ctx.obj["profile"] = profile
    ctx.obj["flash_attention"] = flash_attention
    ctx.obj["tiled_refiner"] = tiled_refiner
    ctx.obj["cache_clearing"] = cache_clearing
    ctx.obj["disable_cudnn_benchmark"] = disable_cudnn_benchmark
    ctx.obj["token_routing"] = token_routing
    ctx.obj["compile_mode"] = compile_mode
    ctx.obj["tensorrt"] = tensorrt
    ctx.obj["tile_size"] = tile_size
    ctx.obj["tile_overlap"] = tile_overlap
    ctx.obj["gpu_postprocess"] = gpu_postprocess
    ctx.obj["comp_format"] = comp_format
    ctx.obj["comp_checkerboard"] = comp_checkerboard
    ctx.obj["dma_buffers"] = dma_buffers
    ctx.obj["precision"] = precision
    ctx.obj["list_only"] = list_only
    logger.info("Using device: %s", ctx.obj["device"])

    # If no subcommand was given (just global flags like --devices 0),
    # launch the TUI.  ctx.invoked_subcommand is None when Typer runs
    # the callback without a subcommand.
    if ctx.invoked_subcommand is None:
        from tui.app import CorridorKeyApp

        CorridorKeyApp().run()
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_path(raw_path: str) -> str:
    """Resolve a Windows or local path to a real local path."""
    if os.path.exists(raw_path):
        return raw_path
    mapped = map_path(raw_path)
    if os.path.exists(mapped):
        return mapped
    console.print(
        f"[bold red]ERROR:[/bold red] Path does not exist: {raw_path}\nExpected Linux Mount Root: {LINUX_MOUNT_ROOT}"
    )
    raise typer.Exit(code=1)


def _propose_structure(path: str) -> None:
    """If the path lacks Input/ or AlphaHint/, offer to create them."""
    has_input = os.path.isdir(os.path.join(path, Dir.INPUT))

    if has_input:
        return

    console.print(f"[yellow]Path '{os.path.basename(path)}' has no Input/ folder.[/yellow]")
    try:
        answer = _readline_input(
            "Create folder structure (Input/, AlphaHint/)?"
            " [bold magenta]\\[[/bold magenta]"
            "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
            "[bold magenta]/[/bold magenta]"
            "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
            "[bold magenta]][/bold magenta]"
            " [cyan](no)[/cyan]",
        ).strip().lower()
    except EOFError:
        answer = "n"

    if answer in ("y", "yes"):
        for subdir in [Dir.INPUT, Dir.ALPHA_HINT]:
            os.makedirs(os.path.join(path, subdir), exist_ok=True)
        console.print("[green]Created Input/, AlphaHint/[/green]")


def _load_clip(path: str) -> ClipEntry:
    """Load a single clip from the given path.

    The path IS the clip root — it should contain Input/, AlphaHint/, etc.
    """
    from .clip_state import ClipEntry

    entry = ClipEntry(name=os.path.basename(path), root_path=path)
    entry.find_assets()
    return entry


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("generate-alphas")
def generate_alphas_cmd(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Clip root path")],
    model: Annotated[
        str,
        typer.Option("--model", help="Alpha generator: gvm, birefnet, videomama"),
    ],
    start: Annotated[
        Optional[int],
        typer.Option("--start", help="Start frame (0-based, inclusive)"),
    ] = None,
    end: Annotated[
        Optional[int],
        typer.Option("--end", help="End frame (0-based, inclusive)"),
    ] = None,
    alpha_mode: Annotated[
        str,
        typer.Option("--alpha-mode", help="replace, fill, or skip when AlphaHint exists"),
    ] = "replace",
    mask_dir: Annotated[
        Optional[str],
        typer.Option("--mask-dir", help="Override VideoMamaMaskHint/ path (videomama only)"),
    ] = None,
) -> None:
    """Generate alpha hints using the specified model.

    [dim]--model gvm|birefnet|videomama  --start N  --end N  --alpha-mode replace|fill|skip[/dim]
    """
    from .api.events import JobCompleted, JobFailed, JobCancelled
    from .api.types import GenerateParams
    from .client import EngineClient

    resolved = _resolve_path(path)
    _propose_structure(resolved)

    # Build frame range string from start/end
    frames = None
    if start is not None or end is not None:
        s = (start or 0) + 1  # convert 0-based to 1-based
        if end is not None:
            frames = f"{s}-{end + 1}"
        else:
            frames = f"{s}-"

    params = GenerateParams(
        path=resolved,
        model=model,
        mode=alpha_mode,
        frames=frames,
        device=ctx.obj.get("device", "auto"),
    )

    client = EngineClient.spawn()
    try:
        job_id = client.submit_generate(params)
        result = _render_events_with_progress(client, job_id)

        if isinstance(result, JobCompleted):
            console.print("[bold green]Alpha generation complete.")
        elif isinstance(result, JobFailed):
            console.print(f"[bold red]Alpha generation failed:[/bold red] {result.error}")
            raise typer.Exit(code=1)
        elif isinstance(result, JobCancelled):
            console.print("[yellow]Alpha generation cancelled.[/yellow]")
        else:
            console.print("[bold red]Alpha generation failed (no response).[/bold red]")
            raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelling...[/yellow]")
        try:
            client.cancel_job(job_id)
        except Exception:
            pass
    except EngineError as e:
        console.print(f"[bold red]Engine error:[/bold red] {e.message}")
        raise typer.Exit(code=1) from e
    finally:
        client.close(shutdown=True)


@app.command()
def inference(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path containing clip folders")],
    start: Annotated[
        Optional[int],
        typer.Option("--start", help="Start frame (0-based, inclusive)"),
    ] = None,
    end: Annotated[
        Optional[int],
        typer.Option("--end", help="End frame (0-based, inclusive)"),
    ] = None,
    linear: Annotated[
        Optional[bool],
        typer.Option("--linear/--srgb", help="Input colorspace (default: prompt)"),
    ] = None,
    despill: Annotated[
        Optional[int],
        typer.Option("--despill", help="Despill strength 0-10 (default: prompt)"),
    ] = None,
    despeckle: Annotated[
        Optional[bool],
        typer.Option("--despeckle/--no-despeckle", help="Auto-despeckle toggle (default: prompt)"),
    ] = None,
    despeckle_size: Annotated[
        Optional[int],
        typer.Option("--despeckle-size", help="Min pixel size for despeckle (default: prompt)"),
    ] = None,
    refiner: Annotated[
        Optional[float],
        typer.Option("--refiner", help="Refiner strength multiplier (default: prompt)"),
    ] = None,
) -> None:
    """Run CorridorKey inference on clips with Input + AlphaHint.

    [dim]--start N  --end N  --linear/--srgb  --despill 0-10  --despeckle/--no-despeckle  --despeckle-size N  --refiner F[/dim]
    """
    from .api.events import JobCompleted, JobFailed, JobCancelled
    from .api.types import InferenceParams, InferenceSettings as APISettings
    from .client import EngineClient

    resolved = _resolve_path(path)
    _propose_structure(resolved)

    # Build frame range
    frames = None
    if start is not None or end is not None:
        s = (start or 0) + 1
        if end is not None:
            frames = f"{s}-{end + 1}"
        else:
            frames = f"{s}-"

    # Build inference settings (prompt or flags)
    required_flags_set = all(v is not None for v in [linear, despill, despeckle, refiner])
    if required_flags_set:
        despill_clamped = max(0, min(10, despill))
        settings = APISettings(
            input_is_linear=linear,
            despill_strength=despill_clamped / 10.0,
            auto_despeckle=despeckle,
            despeckle_size=despeckle_size if despeckle_size is not None else 400,
            refiner_scale=refiner,
        )
    else:
        try:
            pipeline_settings, _ = _prompt_inference_settings(
                default_linear=linear,
                default_despill=despill,
                default_despeckle=despeckle,
                default_despeckle_size=despeckle_size,
                default_refiner=refiner,
            )
        except EOFError:
            console.print("[yellow]Aborted.[/yellow]")
            return
        if pipeline_settings is None:
            console.print("[yellow]Aborted.[/yellow]")
            return
        settings = APISettings(
            input_is_linear=pipeline_settings.input_is_linear,
            despill_strength=pipeline_settings.despill_strength,
            auto_despeckle=pipeline_settings.auto_despeckle,
            despeckle_size=pipeline_settings.despeckle_size,
            refiner_scale=pipeline_settings.refiner_scale,
        )

    opt = _build_optimization_params(ctx.obj)

    params = InferenceParams(
        path=resolved,
        frames=frames,
        device=ctx.obj.get("device", "auto"),
        backend=ctx.obj.get("backend", "auto"),
        settings=settings,
        optimization=opt,
        devices=ctx.obj.get("devices_list"),
        img_size=ctx.obj.get("img_size", 2048),
        read_workers=ctx.obj.get("read_workers", 0),
        write_workers=ctx.obj.get("write_workers", 0),
        cpus=ctx.obj.get("cpus", 0),
        gpu_resilience=ctx.obj.get("gpu_resilience", False),
    )

    client = EngineClient.spawn()
    try:
        job_id = client.submit_inference(params)
        result = _render_events_with_progress(client, job_id)

        if isinstance(result, JobCompleted):
            console.print("[bold green]Inference complete.")
        elif isinstance(result, JobFailed):
            console.print(f"[bold red]Inference failed:[/bold red] {result.error}")
            raise typer.Exit(code=1)
        elif isinstance(result, JobCancelled):
            console.print("[yellow]Inference cancelled.[/yellow]")
        else:
            console.print("[bold red]Inference failed (no response).[/bold red]")
            raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelling...[/yellow]")
        try:
            client.cancel_job(job_id)
        except Exception:
            pass
    except EngineError as e:
        console.print(f"[bold red]Engine error:[/bold red] {e.message}")
        raise typer.Exit(code=1) from e
    finally:
        client.close(shutdown=True)


@app.command("wizard", hidden=True)
def wizard_cmd(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
) -> None:
    """Interactive wizard (default when a bare path is given)."""
    optimization_config = _build_optimization_config(ctx.obj)
    interactive_wizard(
        path,
        device=ctx.obj["device"],
        devices=ctx.obj["devices_list"],
        backend=ctx.obj["backend"],
        optimization_config=optimization_config,
        img_size=ctx.obj["img_size"],
        read_workers=ctx.obj["read_workers"],
        write_workers=ctx.obj["write_workers"],
        cpus=ctx.obj["cpus"],
        gpu_resilience=ctx.obj["gpu_resilience"],
        list_only=ctx.obj.get("list_only", False),
    )


# ---------------------------------------------------------------------------
# Menu renderer — modular panel display and clearing
# ---------------------------------------------------------------------------


class MenuRenderer:
    """Manages rendering and clearing of Rich panels in the terminal.

    Instead of manual ANSI line counting, each ``show()`` call captures
    the rendered output, counts its lines, and records the total so that
    ``clear()`` can erase exactly the right region — including the prompt
    line the user interacted with.
    """

    def __init__(self, con: Console) -> None:
        self._console = con
        self._line_count = 0

    def show(self, *renderables: object) -> None:
        """Render one or more Rich objects and track their line count."""
        with self._console.capture() as cap:
            for r in renderables:
                self._console.print(r)
        output = cap.get()
        self._line_count = output.count("\n")
        sys.stdout.write(output)
        sys.stdout.flush()

    def clear(self, *, extra_lines: int = 1) -> None:
        """Erase the last rendered menu plus *extra_lines* (prompt line).

        By default ``extra_lines=1`` accounts for the single prompt+answer
        line produced by ``input()``.  Pass ``0`` if no prompt was shown.
        """
        total = self._line_count + extra_lines
        if total > 0 and self._console.is_terminal:
            sys.stdout.write(f"\033[{total}F\033[J")
            sys.stdout.flush()
        self._line_count = 0

    def prompt(
        self,
        markup: str,
        valid: set[str],
        *,
        suffix: str = ": ",
    ) -> str:
        """Prompt until user enters a value in *valid*.  Bad inputs are
        erased in-place without affecting the tracked line count."""
        while True:
            try:
                choice = _readline_input(markup, suffix=suffix)
            except EOFError:
                return ""
            choice = choice.strip().lower()
            if choice in valid:
                return choice
            if self._console.is_terminal:
                sys.stdout.write("\033[A\r\033[J")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# Wizard (rich-styled)
# ---------------------------------------------------------------------------


def _prompt_alpha_mode(clip_name: str, frame_count: int, input_count: int, is_temporal: bool = False) -> AlphaMode:
    """Ask the user how to handle existing AlphaHint frames.

    Returns AlphaMode.REPLACE, FILL, or SKIP.
    """
    from .pipeline import AlphaMode

    missing = input_count - frame_count
    has_missing = missing > 0

    console.print(
        f"[yellow]AlphaHint/ already exists[/yellow] "
        f"({frame_count} frame{'s' if frame_count != 1 else ''}"
        f"{f', {missing} missing' if has_missing else ''})."
    )

    choices = (
        " [bold magenta]\\[[/bold magenta]"
        "[bold magenta]r[/bold magenta][magenta]eplace all[/magenta]"
    )
    valid = {"r", "c"}
    if has_missing:
        choices += (
            "[bold magenta]/[/bold magenta]"
            "[bold magenta]f[/bold magenta][magenta]ill missing[/magenta]"
        )
        valid.add("f")
    choices += (
        "[bold magenta]/[/bold magenta]"
        "[bold magenta]c[/bold magenta][magenta]ancel[/magenta]"
        "[bold magenta]][/bold magenta]"
    )

    try:
        answer = _readline_input(choices).strip().lower()
    except EOFError:
        return AlphaMode.SKIP

    if answer in ("r", "replace"):
        return AlphaMode.REPLACE
    if answer in ("f", "fill") and has_missing:
        if is_temporal:
            console.print("[dim]Temporal model — will regenerate all frames for consistency.[/dim]")
        return AlphaMode.FILL
    return AlphaMode.SKIP


def interactive_wizard(
    win_path: str,
    device: str | None = None,
    devices: list[str] | None = None,
    backend: str | None = None,
    optimization_config: object | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    read_workers: int = 0,
    write_workers: int = 0,
    cpus: int = 0,
    gpu_resilience: bool = False,
    list_only: bool = False,
) -> None:
    from .clip_state import ClipEntry
    from .pipeline import (
        AlphaMode,
        generate_alpha_hints,
        run_inference,
    )

    console.print(Panel("[bold]CORRIDOR KEY -- SMART WIZARD[/bold]", style="cyan"))

    # 1. Resolve Path
    if os.path.exists(win_path):
        process_path = win_path
        console.print(f"Path: [bold]{process_path}[/bold]")
    else:
        process_path = map_path(win_path)
        console.print(f"Linux/Remote Path: [bold]{process_path}[/bold]")

        if not os.path.exists(process_path):
            console.print(
                f"\n[bold red]ERROR:[/bold red] Path does not exist locally OR on Linux mount!\n"
                f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}"
            )
            raise typer.Exit(code=1)

    # 2. Propose structure if needed
    _propose_structure(process_path)

    def _scan_clip() -> ClipEntry:
        """Rescan the single clip from disk."""
        entry = ClipEntry(os.path.basename(process_path), process_path)
        with contextlib.suppress(FileNotFoundError, ValueError, OSError):
            entry.find_assets()
        return entry

    # 3. Status Check Loop
    _erase_lines = 0

    def _erase_menu() -> None:
        nonlocal _erase_lines
        if _erase_lines > 0 and console.is_terminal:
            sys.stdout.write(f"\033[{_erase_lines}F\033[J")
            sys.stdout.flush()
        _erase_lines = 0

    try:
        while True:
            clip = _scan_clip()

            # Build status info
            info: list[str] = []
            if clip.input_asset:
                n = clip.input_asset.frame_count
                info.append(f"{n} input frame{'s' if n != 1 else ''}")
            else:
                info.append("[red]no Input[/red]")
            if clip.alpha_asset:
                n = clip.alpha_asset.frame_count
                info.append(f"{n} alpha frame{'s' if n != 1 else ''}")
            has_mask = clip.mask_asset is not None
            if has_mask:
                info.append("mask hint")

            # Determine readiness
            is_ready = False
            warning = None
            if clip.input_asset and clip.alpha_asset:
                if clip.input_asset.frame_count == clip.alpha_asset.frame_count:
                    is_ready = True
                else:
                    warning = (
                        f"Frame mismatch: {clip.input_asset.frame_count} input"
                        f" vs {clip.alpha_asset.frame_count} alpha"
                    )

            table = Table(show_lines=True)
            table.add_column("Clip", style="bold")
            table.add_column("Status")
            table.add_column("Details")

            status = "[green]Ready[/green]" if is_ready else "[yellow]Needs Alpha[/yellow]"
            details = ", ".join(info)
            if warning:
                details += f" [bold yellow]{warning}[/bold yellow]"
            table.add_row(clip.name, status, details)

            if list_only:
                console.print(table)
                return

            actions: list[str] = [
                f"[bold]v[/bold] -- Run VideoMaMa {'[green](mask found)[/green]' if has_mask else '[dim](no mask)[/dim]'}",
                "[bold]g[/bold] -- Run GVM",
                "[bold]b[/bold] -- Run BiRefNet (~4GB VRAM)",
                f"[bold]i[/bold] -- Run Inference {'[green](ready)[/green]' if is_ready else '[dim](not ready)[/dim]'}",
                "[bold]r[/bold] -- Re-scan",
                "[bold]q[/bold] -- Quit [dim](ctrl+d)[/dim]",
            ]

            actions_panel = Panel("\n".join(actions), title="Actions", style="blue")

            _erase_menu()
            with console.capture() as cap:
                console.print(table)
                console.print(actions_panel)
            menu_output = cap.get()
            menu_line_count = menu_output.count("\n")
            sys.stdout.write(menu_output)
            sys.stdout.flush()

            valid_choices = {"v", "g", "b", "i", "r", "q"}
            while True:
                try:
                    choice = _readline_input("Select action")
                except EOFError:
                    choice = "q"
                    break
                choice = choice.strip().lower()
                if choice in valid_choices:
                    break
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

            if choice in ("v", "g", "b"):
                _erase_lines = menu_line_count + 2
                _erase_menu()

                model_map = {"g": "gvm", "b": "birefnet", "v": "videomama"}
                label_map = {"g": "GVM Auto-Matte", "b": "BiRefNet Auto-Matte", "v": "VideoMaMa"}
                model_name = model_map[choice]
                is_temporal = choice in ("g", "v")

                # VideoMaMa: check for mask
                if choice == "v":
                    clip = _scan_clip()
                    if not clip.mask_asset:
                        console.print(
                            "[yellow]No VideoMamaMaskHint/ folder found.[/yellow]\n"
                            "Place a mask sequence or video in [bold]VideoMamaMaskHint/[/bold]."
                        )
                        with contextlib.suppress(EOFError):
                            _readline_input("Press Enter to return to menu", suffix="")
                        continue

                with console.capture() as cap_alpha:
                    console.print(Panel(label_map[choice], style="magenta"))
                alpha_hdr = cap_alpha.get()
                sys.stdout.write(alpha_hdr)
                sys.stdout.flush()

                clip = _scan_clip()
                if clip.alpha_asset:
                    input_count = clip.input_asset.frame_count if clip.input_asset else 0
                    mode = _prompt_alpha_mode(
                        clip.name, clip.alpha_asset.frame_count, input_count, is_temporal=is_temporal
                    )
                    if mode is AlphaMode.SKIP:
                        continue
                else:
                    mode = AlphaMode.REPLACE

                try:
                    succeeded = generate_alpha_hints([clip], model=model_name, device=device, alpha_mode=mode)
                    if succeeded == 0:
                        console.print("[bold yellow]Warning: alpha generation produced no output for this clip.[/bold yellow]")
                except FileNotFoundError as e:
                    console.print(f"[bold red]Model not found:[/bold red] {e}")
                except (ImportError, RuntimeError) as e:
                    console.print(f"[bold red]Model initialization failed:[/bold red] {e}")
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                with contextlib.suppress(EOFError):
                    _readline_input("Press Enter to return to menu", suffix="")

            elif choice == "i":
                _erase_lines = menu_line_count + 2
                _erase_menu()

                # Rescan to get fresh state
                clip = _scan_clip()
                if not clip.input_asset or not clip.alpha_asset:
                    console.print("[yellow]Not ready for inference (need both Input and AlphaHint).[/yellow]")
                    with contextlib.suppress(EOFError):
                        _readline_input("Press Enter to return to menu", suffix="")
                    continue
                if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                    console.print(
                        f"[yellow]Frame count mismatch: {clip.input_asset.frame_count} input"
                        f" vs {clip.alpha_asset.frame_count} alpha.[/yellow]"
                    )
                    with contextlib.suppress(EOFError):
                        _readline_input("Press Enter to return to menu", suffix="")
                    continue

                with console.capture() as cap_i:
                    console.print(Panel("Corridor Key Inference", style="magenta"))
                i_hdr = cap_i.get()
                i_hdr_lines = i_hdr.count("\n")
                sys.stdout.write(i_hdr)
                sys.stdout.flush()
                try:
                    settings, settings_lines = _prompt_inference_settings()
                    if settings is None:
                        _erase_lines = i_hdr_lines + settings_lines + 2
                        continue
                    # Free VRAM from alpha generation before loading inference engine
                    gc.collect()
                    from .device import clear_device_cache
                    clear_device_cache(device or "auto")
                    with ProgressContext() as ctx_progress:
                        run_inference(
                            [clip],
                            device=device,
                            backend=backend,
                            settings=settings,
                            optimization_config=optimization_config,
                            devices=devices,
                            img_size=img_size,
                            read_workers=read_workers,
                            write_workers=write_workers,
                            cpus=cpus,
                            gpu_resilience=gpu_resilience,
                            on_clip_start=ctx_progress.on_clip_start,
                            on_progress=ctx_progress.on_progress,
                        )
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Inference failed:[/bold red] {e}")
                with contextlib.suppress(EOFError):
                    _readline_input("Press Enter to return to menu", suffix="")

            elif choice == "r":
                _erase_lines = menu_line_count + 2
                continue

            elif choice == "q":
                _erase_lines = menu_line_count + 1
                _erase_menu()
                break
    except KeyboardInterrupt:
        pass

    console.print("[bold green]Wizard complete. Goodbye![/bold green]")


@app.command("serve")
def serve_cmd(
    listen: Annotated[
        Optional[str],
        typer.Option("--listen", help="TCP address (e.g. :9400 or 0.0.0.0:9400)"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level"),
    ] = "INFO",
) -> None:
    """Start the JSON-RPC engine server (stdio or TCP daemon)."""
    from ck_engine.engine.server import main as server_main

    # Build sys.argv for the server's argparse
    argv = ["corridorkey-engine serve"]
    if listen:
        argv.extend(["--listen", listen])
    argv.extend(["--log-level", log_level])

    sys.argv = argv
    server_main()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by the ``corridorkey-engine`` console script.

    Routing logic:
      corridorkey-engine                          → launch Textual TUI
      corridorkey-engine /path                    → launch TUI with path pre-set
      corridorkey-engine inference /path [OPTIONS] → headless inference
      corridorkey-engine generate-alphas ...      → headless alpha gen
      corridorkey-engine serve [--listen :9400]   → JSON-RPC server
      corridorkey-engine --help                   → global help
    """
    if len(sys.argv) <= 1:
        # No args → launch TUI
        from tui.app import CorridorKeyApp

        CorridorKeyApp().run()
        return

    first = sys.argv[1]

    # "serve" bypasses Typer entirely — it must not print anything to
    # stdout because the JSON-RPC transport uses stdio.
    if first == "serve":
        from ck_engine.engine.server import main as server_main

        sys.argv = sys.argv[1:]  # strip "serve" so argparse sees --listen etc.
        server_main()
        return

    if first.startswith("-") or first in _SUBCOMMANDS:
        # Known subcommand or flag → Typer handles it
        try:
            app()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(130)
        return

    # Bare path → launch TUI with path pre-set
    from tui.app import CorridorKeyApp

    CorridorKeyApp(initial_path=first).run()
