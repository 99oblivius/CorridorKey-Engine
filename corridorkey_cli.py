"""CorridorKey command-line interface and interactive wizard.

This module handles CLI subcommands, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage:
    uv run corridorkey wizard "V:\\..."
    uv run corridorkey run-inference
    uv run corridorkey generate-alphas
    uv run corridorkey list-clips
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Annotated, Optional, Self

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

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    InferenceSettings,
    generate_alphas,
    generate_alphas_birefnet,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
    scan_clips,
)
from device_utils import resolve_device

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="corridorkey",
    help="Neural network green screen keying for professional VFX pipelines.",
    rich_markup_mode="rich",
    no_args_is_help=True,
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
    ``on_frame_complete(completed, total, bytes_read, bytes_written)``
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

    def on_frame_complete(
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
                gamma_choice = _readline_input(
                    "Input colorspace"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]l[/bold magenta][magenta]inear[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]s[/bold magenta][magenta]rgb[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    " [cyan](srgb)[/cyan]",
                )
                val = gamma_choice.strip().lower()
                if not val:
                    input_is_linear = False
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
                raw = _readline_input(
                    "Despill strength [cyan](0-10, 10 = max despill)[/cyan] [cyan](5)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despill_int = 5
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
                    " [cyan](yes)[/cyan]",
                )
                val = raw.strip().lower()
                if not val or val in ("y", "yes"):
                    auto_despeckle = True
                    lines += 1
                    break
                if val in ("n", "no"):
                    auto_despeckle = False
                    lines += 1
                    break
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

        despeckle_size = default_despeckle_size if default_despeckle_size is not None else 400
        if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
            while True:
                raw = _readline_input(
                    "Despeckle size [cyan](min pixels for a spot)[/cyan] [cyan](400)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despeckle_size = 400
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
                    "Refiner strength multiplier [dim](experimental)[/dim] [cyan](1.0)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    refiner_scale = 1.0
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


def _build_optimization_config(
    backend: str,
    profile: str | None,
    flash_attention: bool | None,
    tiled_refiner: bool | None,
    cache_clearing: bool | None,
    disable_cudnn_benchmark: bool | None,
    token_routing: bool,
    compile_mode: str | None,
    tensorrt: bool,
    tile_size: int,
    tile_overlap: int,
    gpu_postprocess: bool | None,
    no_comp_png: bool,
    comp_checkerboard: bool | None,
    dma_buffers: int | None,
    precision: str,
) -> object | None:
    """Build an OptimizationConfig from CLI args.

    Returns None if no profile or optimization flags were specified.
    """
    from dataclasses import replace

    from CorridorKeyModule.optimization_config import OptimizationConfig

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
            or no_comp_png
            or token_routing
            or tensorrt
            or dma_buffers != 2
            or precision != "fp16"
            or tile_size != 512
            or tile_overlap != 128
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
    if tile_size != 512:
        overrides["tile_size"] = tile_size
    if tile_overlap != 128:
        overrides["tile_overlap"] = tile_overlap
    if gpu_postprocess is not None:
        overrides["gpu_postprocess"] = gpu_postprocess
    if no_comp_png:
        overrides["output_comp_png"] = False
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
    if resolved_precision != "float16":
        overrides["model_precision"] = resolved_precision

    if overrides:
        config = replace(config, **overrides)

    return config


# ---------------------------------------------------------------------------
# Typer callback (shared options)
# ---------------------------------------------------------------------------


@app.callback()
def app_callback(
    ctx: typer.Context,
    device: Annotated[
        str,
        typer.Option(help="Compute device: auto, cuda, mps, cpu"),
    ] = "auto",
) -> None:
    """Neural network green screen keying for professional VFX pipelines."""
    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
    logger.info("Using device: %s", ctx.obj["device"])


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("list-clips")
def list_clips_cmd(ctx: typer.Context) -> None:
    """List all clips in ClipsForInference and their status."""
    scan_clips()


@app.command("generate-alphas")
def generate_alphas_cmd(ctx: typer.Context) -> None:
    """Generate coarse alpha hints via GVM for clips missing them."""
    clips = scan_clips()
    try:
        generate_alphas(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_log_only)
    except KeyboardInterrupt:
        console.print("\n[yellow]Alpha generation interrupted.[/yellow]")
        return
    console.print("[bold green]Alpha generation complete.")


@app.command("generate-alphas-birefnet")
def generate_alphas_birefnet_cmd(ctx: typer.Context) -> None:
    """Generate coarse alpha hints via BiRefNet (~4GB VRAM) for clips missing them."""
    clips = scan_clips()
    try:
        generate_alphas_birefnet(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_log_only)
    except KeyboardInterrupt:
        console.print("\n[yellow]BiRefNet alpha generation interrupted.[/yellow]")
        return
    console.print("[bold green]BiRefNet alpha generation complete.")


@app.command("run-inference")
def run_inference_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option(help="Backend: auto (optimized on CUDA, mlx on Apple Silicon), torch, torch_optimized, mlx"),
    ] = "auto",
    max_frames: Annotated[
        Optional[int],
        typer.Option("--max-frames", help="Limit frames per clip"),
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
    # --- Multi-GPU ---
    devices: Annotated[
        Optional[str],
        typer.Option("--devices", help="Comma-separated GPU indices (e.g. 0,1)"),
    ] = None,
    img_size: Annotated[
        int,
        typer.Option("--img-size", help="Model input resolution"),
    ] = 2048,
    read_workers: Annotated[
        int,
        typer.Option("--read-workers", help="Reader thread pool size (0=auto)"),
    ] = 0,
    write_workers: Annotated[
        int,
        typer.Option("--write-workers", help="Writer thread pool size (0=auto)"),
    ] = 0,
    # --- Optimization ---
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", help="Optimization profile: optimized, original, experimental*, performance*"),
    ] = None,
    flash_attention: Annotated[
        Optional[bool],
        typer.Option("--flash-attention/--no-flash-attention", help="(optimized: on) FlashAttention patching"),
    ] = None,
    tiled_refiner: Annotated[
        Optional[bool],
        typer.Option("--tiled-refiner/--no-tiled-refiner", help="(optimized: on) Tiled CNN refiner"),
    ] = None,
    cache_clearing: Annotated[
        Optional[bool],
        typer.Option("--cache-clearing/--no-cache-clearing", help="(optimized: on) CUDA cache clearing"),
    ] = None,
    disable_cudnn_benchmark: Annotated[
        Optional[bool],
        typer.Option("--cudnn-benchmark/--no-cudnn-benchmark", help="(opt:off) cuDNN auto-tune, faster, +2-5 GB VRAM"),
    ] = None,
    token_routing: Annotated[
        bool,
        typer.Option("--token-routing", help="Experimental token routing (improves speed at high res)", is_flag=True),
    ] = False,
    compile_mode: Annotated[
        Optional[str],
        typer.Option("--compile-mode", help="torch.compile mode: default, reduce-overhead*, max-autotune*"),
    ] = None,
    tensorrt: Annotated[
        bool,
        typer.Option("--tensorrt", help="TensorRT compilation (broken, WIP)", is_flag=True),
    ] = False,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="Tile size for tiled refiner"),
    ] = 512,
    tile_overlap: Annotated[
        int,
        typer.Option("--tile-overlap", help="Tile overlap in pixels"),
    ] = 128,
    gpu_postprocess: Annotated[
        Optional[bool],
        typer.Option("--gpu-postprocess/--cpu-postprocess", help="Postprocessing device (GPU default, +~1.5 GB VRAM)"),
    ] = None,
    no_comp_png: Annotated[
        bool,
        typer.Option("--no-comp-png", help="Disable transparent RGBA composite PNG output", is_flag=True),
    ] = False,
    comp_checkerboard: Annotated[
        bool,
        typer.Option("--checkerboard", help="Checkerboard background for composite (not transparent)", is_flag=True),
    ] = False,
    dma_buffers: Annotated[
        int,
        typer.Option("--dma-buffers", help="Pinned DMA buffer count 2-3"),
    ] = 2,
    precision: Annotated[
        str,
        typer.Option("--precision", help="Model weight precision: fp16, bf16, fp32"),
    ] = "fp16",
) -> None:
    """Run CorridorKey inference on clips with Input + AlphaHint.

    Settings can be passed as flags for non-interactive use, or omitted to
    prompt interactively.
    """
    clips = scan_clips()

    # Parse devices
    devices_list = None
    if devices:
        devices_list = [f"cuda:{idx.strip()}" for idx in devices.split(",")]
        logger.info("Multi-GPU devices: %s", devices_list)

    # Build optimization config
    optimization_config = _build_optimization_config(
        backend=backend,
        profile=profile,
        flash_attention=flash_attention,
        tiled_refiner=tiled_refiner,
        cache_clearing=cache_clearing,
        disable_cudnn_benchmark=disable_cudnn_benchmark,
        token_routing=token_routing,
        compile_mode=compile_mode,
        tensorrt=tensorrt,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        gpu_postprocess=gpu_postprocess,
        no_comp_png=no_comp_png,
        comp_checkerboard=comp_checkerboard,
        dma_buffers=dma_buffers,
        precision=precision,
    )
    if optimization_config is not None:
        logger.info(optimization_config.summary())

    # Build inference settings (prompt or flags)
    required_flags_set = all(v is not None for v in [linear, despill, despeckle, refiner])
    if required_flags_set:
        assert linear is not None
        assert despill is not None
        assert despeckle is not None
        assert refiner is not None
        despill_clamped = max(0, min(10, despill))
        settings = InferenceSettings(
            input_is_linear=linear,
            despill_strength=despill_clamped / 10.0,
            auto_despeckle=despeckle,
            despeckle_size=despeckle_size if despeckle_size is not None else 400,
            refiner_scale=refiner,
        )
    else:
        try:
            settings, _ = _prompt_inference_settings(
                default_linear=linear,
                default_despill=despill,
                default_despeckle=despeckle,
                default_despeckle_size=despeckle_size,
                default_refiner=refiner,
            )
        except EOFError:
            console.print("[yellow]Aborted.[/yellow]")
            return
        if settings is None:
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        with ProgressContext() as ctx_progress:
            run_inference(
                clips,
                device=ctx.obj["device"],
                backend=backend,
                max_frames=max_frames,
                settings=settings,
                optimization_config=optimization_config,
                devices=devices_list,
                img_size=img_size,
                read_workers=read_workers,
                write_workers=write_workers,
                on_clip_start=ctx_progress.on_clip_start,
                on_frame_complete=ctx_progress.on_frame_complete,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Inference interrupted.[/yellow]")
        return

    console.print("[bold green]Inference complete.")


@app.command()
def wizard(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
    backend: Annotated[
        str,
        typer.Option(help="Backend: auto (optimized on CUDA, mlx on Apple Silicon), torch, torch_optimized, mlx"),
    ] = "auto",
    # --- Multi-GPU ---
    devices: Annotated[
        Optional[str],
        typer.Option("--devices", help="Comma-separated GPU indices (e.g. 0,1)"),
    ] = None,
    img_size: Annotated[
        int,
        typer.Option("--img-size", help="Model input resolution"),
    ] = 2048,
    read_workers: Annotated[
        int,
        typer.Option("--read-workers", help="Reader thread pool size (0=auto)"),
    ] = 0,
    write_workers: Annotated[
        int,
        typer.Option("--write-workers", help="Writer thread pool size (0=auto)"),
    ] = 0,
    # --- Optimization ---
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", help="Optimization profile: optimized, original, experimental*, performance*"),
    ] = None,
    flash_attention: Annotated[
        Optional[bool],
        typer.Option("--flash-attention/--no-flash-attention", help="(optimized: on) FlashAttention patching"),
    ] = None,
    tiled_refiner: Annotated[
        Optional[bool],
        typer.Option("--tiled-refiner/--no-tiled-refiner", help="(optimized: on) Tiled CNN refiner"),
    ] = None,
    cache_clearing: Annotated[
        Optional[bool],
        typer.Option("--cache-clearing/--no-cache-clearing", help="(optimized: on) CUDA cache clearing"),
    ] = None,
    disable_cudnn_benchmark: Annotated[
        Optional[bool],
        typer.Option("--cudnn-benchmark/--no-cudnn-benchmark", help="(opt:off) cuDNN auto-tune, faster, +2-5 GB VRAM"),
    ] = None,
    token_routing: Annotated[
        bool,
        typer.Option("--token-routing", help="Experimental token routing (improves speed at high res)", is_flag=True),
    ] = False,
    compile_mode: Annotated[
        Optional[str],
        typer.Option("--compile-mode", help="torch.compile mode: default, reduce-overhead*, max-autotune*"),
    ] = None,
    tensorrt: Annotated[
        bool,
        typer.Option("--tensorrt", help="TensorRT compilation (broken, WIP)", is_flag=True),
    ] = False,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="Tile size for tiled refiner"),
    ] = 512,
    tile_overlap: Annotated[
        int,
        typer.Option("--tile-overlap", help="Tile overlap in pixels"),
    ] = 128,
    gpu_postprocess: Annotated[
        Optional[bool],
        typer.Option("--gpu-postprocess/--cpu-postprocess", help="Postprocessing device (GPU default, +~1.5 GB VRAM)"),
    ] = None,
    no_comp_png: Annotated[
        bool,
        typer.Option("--no-comp-png", help="Disable transparent RGBA composite PNG output", is_flag=True),
    ] = False,
    comp_checkerboard: Annotated[
        bool,
        typer.Option("--checkerboard", help="Checkerboard background for composite (not transparent)", is_flag=True),
    ] = False,
    dma_buffers: Annotated[
        int,
        typer.Option("--dma-buffers", help="Pinned DMA buffer count 2-3"),
    ] = 2,
    precision: Annotated[
        str,
        typer.Option("--precision", help="Model weight precision: fp16, bf16, fp32"),
    ] = "fp16",
) -> None:
    """Interactive wizard for organizing clips and running the pipeline."""
    # Parse devices
    devices_list = None
    if devices:
        devices_list = [f"cuda:{idx.strip()}" for idx in devices.split(",")]

    # Build optimization config
    optimization_config = _build_optimization_config(
        backend=backend,
        profile=profile,
        flash_attention=flash_attention,
        tiled_refiner=tiled_refiner,
        cache_clearing=cache_clearing,
        disable_cudnn_benchmark=disable_cudnn_benchmark,
        token_routing=token_routing,
        compile_mode=compile_mode,
        tensorrt=tensorrt,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        gpu_postprocess=gpu_postprocess,
        no_comp_png=no_comp_png,
        comp_checkerboard=comp_checkerboard,
        dma_buffers=dma_buffers,
        precision=precision,
    )

    interactive_wizard(
        path,
        device=ctx.obj["device"],
        devices=devices_list,
        backend=backend,
        optimization_config=optimization_config,
        img_size=img_size,
        read_workers=read_workers,
        write_workers=write_workers,
    )


# ---------------------------------------------------------------------------
# Wizard (rich-styled)
# ---------------------------------------------------------------------------


def interactive_wizard(
    win_path: str,
    device: str | None = None,
    devices: list[str] | None = None,
    backend: str | None = None,
    optimization_config: object | None = None,
    img_size: int = 2048,
    read_workers: int = 0,
    write_workers: int = 0,
) -> None:
    console.print(Panel("[bold]CORRIDOR KEY -- SMART WIZARD[/bold]", style="cyan"))

    # 1. Resolve Path
    console.print(f"Windows Path: {win_path}")

    if os.path.exists(win_path):
        process_path = win_path
        console.print(f"Running locally: [bold]{process_path}[/bold]")
    else:
        process_path = map_path(win_path)
        console.print(f"Linux/Remote Path: [bold]{process_path}[/bold]")

        if not os.path.exists(process_path):
            console.print(
                f"\n[bold red]ERROR:[/bold red] Path does not exist locally OR on Linux mount!\n"
                f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}"
            )
            raise typer.Exit(code=1)

    # 2. Analyze -- shot or project?
    target_is_shot = False
    if os.path.exists(os.path.join(process_path, "Input")) or glob.glob(os.path.join(process_path, "Input.*")):
        target_is_shot = True

    work_dirs: list[str] = []
    excluded_dirs = {"Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"}
    if target_is_shot:
        work_dirs = [process_path]
    else:
        work_dirs = [
            os.path.join(process_path, d)
            for d in os.listdir(process_path)
            if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
        ]

    console.print(f"\nFound [bold]{len(work_dirs)}[/bold] potential clip folders.")

    # Filter known names from loose video detection
    known_names = {"input", "alphahint", "videomamamaskhint"}
    loose_videos = [
        f
        for f in os.listdir(process_path)
        if is_video_file(f)
        and os.path.isfile(os.path.join(process_path, f))
        and os.path.splitext(f)[0].lower() not in known_names
    ]

    dirs_needing_org = []
    for d in work_dirs:
        has_input = os.path.exists(os.path.join(d, "Input")) or glob.glob(os.path.join(d, "Input.*"))
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))
        if not has_input or not has_alpha or not has_mask:
            dirs_needing_org.append(d)

    if loose_videos or dirs_needing_org:
        if loose_videos:
            console.print(f"Found [yellow]{len(loose_videos)}[/yellow] loose video files:")
            for v in loose_videos:
                console.print(f"  [dim]{v}[/dim]")

        if dirs_needing_org:
            console.print(f"Found [yellow]{len(dirs_needing_org)}[/yellow] folders needing setup:")
            display_limit = 10
            for d in dirs_needing_org[:display_limit]:
                console.print(f"  [dim]{os.path.basename(d)}[/dim]")
            if len(dirs_needing_org) > display_limit:
                console.print(f"  ...and {len(dirs_needing_org) - display_limit} others.")

        # 3. Organize
        try:
            organize = (
                _readline_input(
                    "\nOrganize clips & create hint folders?"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    " [cyan](no)[/cyan]",
                )
                .strip()
                .lower()
            )
        except EOFError:
            organize = "n"

        if organize in ("y", "yes"):
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(process_path, clip_name)

                if os.path.exists(target_folder):
                    logger.warning("Skipping loose video '%s': Target folder '%s' already exists.", v, clip_name)
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(process_path, v), target_file)
                    logger.info("Organized: Moved '%s' to '%s/Input%s'", v, clip_name, ext)
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logger.error("Failed to organize video '%s': %s", v, e)

            for d in work_dirs:
                organize_target(d)
            console.print("[green]Organization complete.[/green]")

            if not target_is_shot:
                work_dirs = [
                    os.path.join(process_path, d)
                    for d in os.listdir(process_path)
                    if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
                ]

    # 4. Status Check Loop
    _erase_lines = 0  # Lines to erase before next menu draw

    def _erase_menu() -> None:
        nonlocal _erase_lines
        if _erase_lines > 0 and console.is_terminal:
            # Move cursor up N lines and clear from cursor to end of screen
            sys.stdout.write(f"\033[{_erase_lines}F\033[J")
            sys.stdout.flush()
        _erase_lines = 0

    try:
        while True:
            ready: list[ClipEntry] = []
            masked: list[ClipEntry] = []
            raw: list[ClipEntry] = []

            for d in work_dirs:
                entry = ClipEntry(os.path.basename(d), d)
                with contextlib.suppress(FileNotFoundError, ValueError, OSError):
                    entry.find_assets()

                has_mask = False
                try:
                    mask_dir = os.path.join(d, "VideoMamaMaskHint")
                    if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                        has_mask = True
                    if not has_mask:
                        for f in os.listdir(d):
                            stem, _ = os.path.splitext(f)
                            if stem.lower() == "videomamamaskhint" and is_video_file(f):
                                has_mask = True
                                break
                except OSError:
                    pass

                if entry.alpha_asset:
                    ready.append(entry)
                elif has_mask:
                    masked.append(entry)
                else:
                    raw.append(entry)

            missing_alpha = masked + raw

            # Build menu renderables
            table = Table(show_lines=True)
            table.add_column("Category", style="bold")
            table.add_column("Count", justify="right")
            table.add_column("Clips")

            table.add_row(
                "[green]Ready[/green] (AlphaHint)",
                str(len(ready)),
                ", ".join(c.name for c in ready) or "---",
            )
            table.add_row(
                "[yellow]Masked[/yellow] (VideoMaMaMaskHint)",
                str(len(masked)),
                ", ".join(c.name for c in masked) or "---",
            )
            table.add_row(
                "[red]Raw[/red] (Input only)",
                str(len(raw)),
                ", ".join(c.name for c in raw) or "---",
            )

            actions: list[str] = []
            if missing_alpha:
                actions.append(f"[bold]v[/bold] -- Run VideoMaMa ({len(masked)} with masks)")
                actions.append(f"[bold]g[/bold] -- Run GVM (auto-matte {len(raw)} clips)")
                actions.append(f"[bold]b[/bold] -- Run BiRefNet (lightweight auto-matte {len(raw)} clips, ~4GB VRAM)")
            if ready:
                actions.append(f"[bold]i[/bold] -- Run Inference ({len(ready)} ready clips)")
            actions.append("[bold]r[/bold] -- Re-scan folders")
            actions.append("[bold]q[/bold] -- Quit [dim](ctrl+d)[/dim]")

            actions_panel = Panel("\n".join(actions), title="Actions", style="blue")

            # Erase previous menu, then render new one
            _erase_menu()
            with console.capture() as cap:
                console.print(table)
                console.print(actions_panel)
            menu_output = cap.get()
            menu_line_count = menu_output.count("\n")
            sys.stdout.write(menu_output)
            sys.stdout.flush()

            # Prompt on a separate line with readline-safe input
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
                # Invalid or empty -- erase the prompt line and re-prompt
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

            if choice == "v":
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_v:
                    console.print(Panel("VideoMaMa", style="magenta"))
                v_hdr = cap_v.get()
                v_hdr_lines = v_hdr.count("\n")
                sys.stdout.write(v_hdr)
                sys.stdout.flush()
                try:
                    run_videomama(missing_alpha, chunk_size=50, device=device)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                try:
                    _readline_input("Press Enter to return to menu", suffix="")
                except EOFError:
                    _erase_lines = v_hdr_lines + 3
                    continue

            elif choice == "g":
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_g:
                    console.print(Panel("GVM Auto-Matte", style="magenta"))
                g_hdr = cap_g.get()
                g_lines = g_hdr.count("\n")
                sys.stdout.write(g_hdr)
                sys.stdout.flush()
                with console.capture() as cap_g2:
                    console.print(f"Will generate alphas for {len(raw)} clips without mask hints.")
                g_info = cap_g2.get()
                g_lines += g_info.count("\n")
                sys.stdout.write(g_info)
                sys.stdout.flush()
                try:
                    gvm_yes = (
                        _readline_input(
                            "Proceed with GVM?"
                            " [bold magenta]\\[[/bold magenta]"
                            "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                            "[bold magenta]/[/bold magenta]"
                            "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                            "[bold magenta]][/bold magenta]"
                            " [cyan](no)[/cyan]",
                        )
                        .strip()
                        .lower()
                    )
                    if gvm_yes in ("y", "yes"):
                        try:
                            generate_alphas(raw, device=device)
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Interrupted.[/yellow]")
                        with contextlib.suppress(EOFError):
                            _readline_input("Press Enter to return to menu", suffix="")
                    else:
                        # Declined -- erase the sub-menu
                        _erase_lines = g_lines + 3
                        continue
                except EOFError:
                    _erase_lines = g_lines + 2
                    continue

            elif choice == "b":
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_b:
                    console.print(Panel("BiRefNet Auto-Matte", style="magenta"))
                b_hdr = cap_b.get()
                b_lines = b_hdr.count("\n")
                sys.stdout.write(b_hdr)
                sys.stdout.flush()
                with console.capture() as cap_b2:
                    console.print(f"Will generate alphas for {len(raw)} clips using BiRefNet (~4GB VRAM).")
                b_info = cap_b2.get()
                b_lines += b_info.count("\n")
                sys.stdout.write(b_info)
                sys.stdout.flush()
                try:
                    biref_yes = (
                        _readline_input(
                            "Proceed with BiRefNet?"
                            " [bold magenta]\\[[/bold magenta]"
                            "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                            "[bold magenta]/[/bold magenta]"
                            "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                            "[bold magenta]][/bold magenta]"
                            " [cyan](no)[/cyan]",
                        )
                        .strip()
                        .lower()
                    )
                    if biref_yes in ("y", "yes"):
                        try:
                            generate_alphas_birefnet(raw, device=device)
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Interrupted.[/yellow]")
                        with contextlib.suppress(EOFError):
                            _readline_input("Press Enter to return to menu", suffix="")
                    else:
                        # Declined -- erase the sub-menu
                        _erase_lines = b_lines + 3
                        continue
                except EOFError:
                    _erase_lines = b_lines + 2
                    continue

            elif choice == "i":
                # Erase the menu before showing inference settings
                _erase_lines = menu_line_count + 2
                _erase_menu()
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
                    with ProgressContext() as ctx_progress:
                        run_inference(
                            ready,
                            device=device,
                            backend=backend,
                            settings=settings,
                            optimization_config=optimization_config,
                            devices=devices,
                            img_size=img_size,
                            read_workers=read_workers,
                            write_workers=write_workers,
                            on_clip_start=ctx_progress.on_clip_start,
                            on_frame_complete=ctx_progress.on_frame_complete,
                        )
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Inference failed:[/bold red] {e}")
                with contextlib.suppress(EOFError):
                    _readline_input("Press Enter to return to menu", suffix="")

            elif choice == "r":
                # Erase menu + prompt line, redraw with fresh scan
                _erase_lines = menu_line_count + 2
                continue

            elif choice == "q":
                # Erase menu + prompt line before goodbye
                _erase_lines = menu_line_count + 1
                _erase_menu()
                break
    except KeyboardInterrupt:
        pass  # Fall through to goodbye message

    console.print("[bold green]Wizard complete. Goodbye![/bold green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by the ``corridorkey`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
