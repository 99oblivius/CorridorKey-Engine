"""Optimization configuration and performance metrics for CorridorKey inference.

Provides an ``OptimizationConfig`` frozen dataclass that toggles each VRAM
optimization independently, and a ``PerformanceMetrics`` helper that can
measure per-stage timing and VRAM usage.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Optimization Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationConfig:
    """Per-optimization toggles and parameters for CorridorKey inference.

    All toggles default to ``False`` so that ``OptimizationConfig()`` produces
    an exact replica of the original (unoptimized) behaviour.  Use the class
    method profiles for common presets.  Individual flags can be overridden
    with ``dataclasses.replace(config, flag=value)``.
    """

    # --- Optimization toggles ---

    flash_attention: bool = False
    """Monkey-patch Hiera global-attention blocks to produce contiguous 4-D
    Q/K/V tensors, enabling PyTorch FlashAttention / Memory-Efficient
    Attention instead of the math fallback that materialises the full
    N x N attention matrix."""

    tiled_refiner: bool = False
    """Process the CNN refiner in overlapping tiles instead of at full
    2048 x 2048 resolution.  Mathematically lossless given the 128 px
    overlap exceeds the refiner's ~65 px receptive field."""

    disable_cudnn_benchmark: bool = False
    """Set ``torch.backends.cudnn.benchmark = False`` to prevent cuDNN from
    allocating workspace memory for benchmark runs (saves 2-5 GB)."""

    cache_clearing: bool = False
    """Call ``torch.cuda.empty_cache()`` between encoder -> decoder and
    decoder -> refiner stages to release intermediate CUDA allocations."""

    mixed_precision: bool = False
    """Enable ``torch.autocast`` with float16 during inference.  When the
    model is already stored in float16 (``model_precision=torch.float16``),
    autocast is redundant and automatically disabled."""

    model_precision: str = "float32"
    """Dtype string for model weights: ``'float32'``, ``'float16'``, or
    ``'bfloat16'``.  Stored as a string because ``torch.dtype`` is not
    natively serialisable.  Use :attr:`model_dtype` for the resolved type."""

    high_matmul_precision: bool = False
    """Set ``torch.set_float32_matmul_precision('high')`` to use TF32
    matmul on Ampere+ GPUs.  Negligible quality impact, measurable speedup."""

    token_routing: bool = False
    """Route 'easy' (solid FG/BG) tokens to a lightweight LTRM module
    instead of global attention.  **Experimental** -- requires fine-tuning;
    disabled by default because the trained attention weights expect all
    tokens to participate."""

    compile_mode: str = "default"
    """``torch.compile`` mode passed as the ``mode=`` argument.  Valid
    values: ``'default'``, ``'reduce-overhead'``, ``'max-autotune'``,
    ``'max-autotune-no-cudagraphs'``.  Higher modes increase first-run
    compilation time but produce faster kernels.  The FX graph cache
    stores autotuned results so the cost is paid once."""

    cuda_graphs: bool = False
    """Enable manual CUDA graph capture for the model forward pass.
    Requires fixed input shapes (satisfied by ``img_size``) and is
    incompatible with ``cache_clearing`` (automatically disabled).
    Only effective on CUDA devices with 12+ GB VRAM."""

    tensorrt: bool = False
    """Use TensorRT via ``torch_tensorrt.compile()`` for optimized
    inference with fused kernels and tensor core utilization.  Requires
    the ``torch-tensorrt`` package.  Falls back to ``torch.compile`` on
    failure.  Incompatible with ``cuda_graphs`` (TensorRT manages its
    own graph capture internally)."""

    # --- Tiling parameters ---

    tile_size: int = 512
    """Tile dimension (square) for the tiled refiner."""

    tile_overlap: int = 128
    """Overlap in pixels between adjacent tiles.  Must exceed the refiner's
    receptive field (~65 px) for lossless tiling."""

    # --- Token routing parameters ---

    edge_threshold_low: float = 0.02
    """Alpha-hint values below this are considered 'easy' background tokens."""

    edge_threshold_high: float = 0.98
    """Alpha-hint values above this are considered 'easy' foreground tokens."""

    min_edge_tokens: int = 64
    """Minimum number of edge tokens; if fewer are found, all tokens fall
    back to full global attention."""

    # --- Performance metrics ---

    enable_metrics: bool = False
    """When ``True``, collect per-stage timing and VRAM usage metrics.
    Results are returned in the ``"metrics"`` key of the output dict."""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def model_dtype(self) -> torch.dtype:
        """Resolve :attr:`model_precision` string to a ``torch.dtype``."""
        _MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return _MAP.get(self.model_precision, torch.float32)

    @property
    def effective_cache_clearing(self) -> bool:
        """Whether ``empty_cache()`` calls should actually run.

        Cache clearing is force-disabled when CUDA graphs are active
        (either manual or via ``max-autotune`` / ``reduce-overhead``
        compile modes) because ``cudaFree`` is illegal inside a
        captured graph.
        """
        if self.cuda_graphs or self.compile_mode in ("max-autotune", "reduce-overhead"):
            return False
        return self.cache_clearing

    @property
    def effective_mixed_precision(self) -> bool:
        """Whether autocast should actually be enabled.

        Autocast is redundant (and slower) when the model weights are
        already in float16, so it is automatically disabled in that case.
        """
        if self.model_precision == "float16":
            return False
        return self.mixed_precision

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    @classmethod
    def original(cls) -> OptimizationConfig:
        """Original behaviour -- no optimizations, no metrics."""
        return cls()

    @classmethod
    def optimized(cls) -> OptimizationConfig:
        """Standard optimized profile: flash_attention + tiled_refiner +
        disable_cudnn_benchmark + cache_clearing + fp16 model + TF32 matmul.
        No token routing."""
        return cls(
            flash_attention=True,
            tiled_refiner=True,
            disable_cudnn_benchmark=True,
            cache_clearing=True,
            mixed_precision=True,
            model_precision="float16",
            high_matmul_precision=True,
            token_routing=False,
        )

    @classmethod
    def experimental(cls) -> OptimizationConfig:
        """All optimizations enabled, including token routing."""
        return cls(
            flash_attention=True,
            tiled_refiner=True,
            disable_cudnn_benchmark=True,
            cache_clearing=True,
            mixed_precision=True,
            model_precision="float16",
            high_matmul_precision=True,
            token_routing=True,
        )

    @classmethod
    def performance(cls) -> OptimizationConfig:
        """Maximum throughput -- disables cache clearing, enables
        autotuning and CUDA graphs.  Requires 12+ GB VRAM at 2048x2048.
        Uses full-resolution refiner and cuDNN benchmarking for fastest
        kernel selection."""
        return cls(
            flash_attention=True,
            tiled_refiner=False,
            disable_cudnn_benchmark=False,
            cache_clearing=False,
            mixed_precision=True,
            model_precision="float16",
            high_matmul_precision=True,
            token_routing=False,
            compile_mode="max-autotune",
        )

    @classmethod
    def from_profile(cls, name: str) -> OptimizationConfig:
        """Resolve a named profile string to an ``OptimizationConfig``.

        Valid names: ``'original'``, ``'optimized'``, ``'experimental'``.
        Raises ``ValueError`` for unknown names.
        """
        profiles: dict[str, classmethod] = {
            "original": cls.original,
            "optimized": cls.optimized,
            "experimental": cls.experimental,
            "performance": cls.performance,
        }
        if name not in profiles:
            raise ValueError(f"Unknown optimization profile '{name}'. Valid profiles: {', '.join(profiles)}")
        return profiles[name]()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def active_optimizations(self) -> list[str]:
        """Return a list of human-readable names for the active optimizations."""
        names: list[str] = []
        if self.flash_attention:
            names.append("flash_attention")
        if self.tiled_refiner:
            names.append(f"tiled_refiner({self.tile_size}x{self.tile_size}/{self.tile_overlap}px)")
        if self.disable_cudnn_benchmark:
            names.append("disable_cudnn_benchmark")
        if self.cache_clearing:
            names.append("cache_clearing")
        if self.model_precision != "float32":
            names.append(f"model_{self.model_precision}")
        if self.effective_mixed_precision:
            names.append("mixed_precision")
        if self.high_matmul_precision:
            names.append("tf32_matmul")
        if self.token_routing:
            names.append(f"token_routing(edge={self.edge_threshold_low}-{self.edge_threshold_high})")
        if self.compile_mode != "default":
            names.append(f"compile_{self.compile_mode}")
        if self.cuda_graphs:
            names.append("cuda_graphs")
        if self.tensorrt:
            names.append("tensorrt")
        return names

    def summary(self) -> str:
        """One-line summary of active optimizations."""
        active = self.active_optimizations()
        if not active:
            return "OptimizationConfig: original (no optimizations)"
        return f"OptimizationConfig: {', '.join(active)}"


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


@dataclass
class StageMetric:
    """Timing and VRAM data for a single inference stage."""

    name: str
    duration_ms: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_peak_mb: float = 0.0


@dataclass
class PerformanceMetrics:
    """Collected per-frame performance data.

    Use the :meth:`measure` context manager to instrument individual stages.
    """

    stages: list[StageMetric] = field(default_factory=list)
    total_duration_ms: float = 0.0
    peak_vram_mb: float = 0.0

    @contextmanager
    def measure(self, stage_name: str, device: torch.device) -> Generator[StageMetric, None, None]:
        """Context manager that records timing and VRAM for *stage_name*.

        Example::

            with metrics.measure("encoder", self.device):
                features = self.model.encoder(x)
        """
        metric = StageMetric(name=stage_name)

        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            metric.vram_before_mb = torch.cuda.memory_allocated(device) / (1024**2)

        t0 = time.perf_counter()
        yield metric
        elapsed = time.perf_counter() - t0

        metric.duration_ms = elapsed * 1000.0

        if is_cuda:
            torch.cuda.synchronize(device)
            metric.vram_after_mb = torch.cuda.memory_allocated(device) / (1024**2)
            metric.vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        self.stages.append(metric)

    def finalize(self, device: torch.device) -> None:
        """Compute totals after all stages have been measured."""
        self.total_duration_ms = sum(s.duration_ms for s in self.stages)
        if device.type == "cuda":
            self.peak_vram_mb = max((s.vram_peak_mb for s in self.stages), default=0.0)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = ["Performance Metrics:"]
        for s in self.stages:
            line = f"  {s.name}: {s.duration_ms:.1f} ms"
            if s.vram_peak_mb > 0:
                line += f"  |  VRAM: {s.vram_before_mb:.0f} -> {s.vram_after_mb:.0f} MB (peak {s.vram_peak_mb:.0f} MB)"
            lines.append(line)
        total_line = f"  TOTAL: {self.total_duration_ms:.1f} ms"
        if self.peak_vram_mb > 0:
            total_line += f"  |  Peak VRAM: {self.peak_vram_mb:.0f} MB"
        lines.append(total_line)
        return "\n".join(lines)
