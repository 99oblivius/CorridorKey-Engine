"""Shared base class for CorridorKey inference engines.

All engine variants (original, optimized, MLX adapter) share the same
``process_frame()`` pipeline and checkpoint-loading logic.  Subclasses
only need to implement :meth:`_create_model` to return the appropriate
model variant.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .constants import (
    DEFAULT_CHECKER_COLOR1,
    DEFAULT_CHECKER_COLOR2,
    DEFAULT_CHECKER_SIZE,
    DEFAULT_DESPECKLE_SIZE,
    DEFAULT_DESPILL_STRENGTH,
    DEFAULT_IMG_SIZE,
    DEFAULT_MATTE_BLUR,
    DEFAULT_MATTE_DILATION,
    DEFAULT_REFINER_SCALE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .core import color_utils as cu
from .optimization_config import OptimizationConfig

logger = logging.getLogger(__name__)


class _GreenFormerTRT(nn.Module):
    """Wrapper that returns a flat tuple for TensorRT compatibility.

    TensorRT requires tensor outputs, not dicts.  This wrapper converts
    the dict return to a tuple, and :meth:`_run_forward` converts it back.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["alpha"], out["fg"]


@dataclasses.dataclass
class PendingTransfer:
    """Handle for a deferred GPU→CPU DMA transfer.

    Returned by ``_postprocess_gpu(sync=False)``.  Call :meth:`resolve`
    to block until the DMA completes and get the numpy result dict.

    Each transfer records a CUDA event immediately after its DMA is
    enqueued on the copy stream.  ``resolve()`` waits on that specific
    event rather than synchronizing the entire stream.

    The transfer also holds a ``threading.Event`` (``_buf_released``)
    that is set after ``resolve()`` copies data out of the pinned
    buffer.  The engine waits on this before reusing that buffer slot,
    guaranteeing the DMA data is never overwritten prematurely.
    """

    _event: torch.cuda.Event | None
    _pinned_buf: torch.Tensor | None
    _cpu_result: dict[str, np.ndarray] | None  # set for CPU fallback
    _buf_released: threading.Event | None  # signalled after numpy .copy()
    _gpu_bulk: torch.Tensor | None = None  # prevents caching allocator reuse during DMA
    _comp_channels: int = 3  # 3 for opaque checkerboard comp, 4 for transparent RGBA

    def resolve(self) -> dict[str, np.ndarray]:
        """Block until this transfer's DMA completes, return numpy arrays."""
        if self._cpu_result is not None:
            return self._cpu_result
        self._event.synchronize()
        # Release GPU bulk tensor — DMA is done, caching allocator can reuse
        self._gpu_bulk = None
        bulk_np = self._pinned_buf.numpy()
        cc = self._comp_channels
        result = {
            "alpha": bulk_np[:, :, 0:1].copy(),
            "fg": bulk_np[:, :, 1:4].copy(),
            "comp": bulk_np[:, :, 4 : 4 + cc].copy(),
            "processed": bulk_np[:, :, 4 + cc : 4 + cc + 4].copy(),
        }
        # Signal that this buffer slot can be reused
        if self._buf_released is not None:
            self._buf_released.set()
        return result


class _BaseCorridorKeyEngine(ABC):
    """Shared inference engine logic for all Torch-based backends.

    Subclasses must implement :meth:`_create_model` to return the
    appropriate ``GreenFormer`` variant.  Everything else -- constructor
    setup, checkpoint loading, ``process_frame()`` -- lives here.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = DEFAULT_IMG_SIZE,
        use_refiner: bool = True,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.config = optimization_config or OptimizationConfig()

        # ImageNet normalization constants (keep on device for GPU preprocessing)
        self.mean_np = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std_np = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

        # GPU-side normalization constants (created after model load)
        self._mean_t: torch.Tensor | None = None
        self._std_t: torch.Tensor | None = None

        # Cached checkerboard — single-entry cache keyed on (w, h).
        # A dict is intentionally NOT used here: storing one entry per unique
        # resolution would grow without bound in mixed-resolution workflows,
        # leaking VRAM/RAM indefinitely.  Instead we keep only the most-recently-
        # used resolution and evict when it changes.
        self._checker_cache_key: tuple[int, int] | None = None
        self._checker_cache_val: torch.Tensor | None = None
        self._checker_cache_cpu_key: tuple[int, int] | None = None
        self._checker_cache_cpu_val: np.ndarray | None = None

        # Dedicated copy stream for GPU→CPU transfers (avoids GIL contention
        # in multi-GPU setups).  Compute runs on the default stream; D2H
        # copies run on _copy_stream so stream.synchronize() only waits for
        # the copy, releasing the GIL without blocking other GPUs' compute.
        self._copy_stream: torch.cuda.Stream | None = None
        # Triple-buffered pinned-memory for D2H transfer.  Three buffers
        # give the drain worker time to resolve (copy out) a frame's data
        # before its buffer slot is reused.  Each slot has a release event
        # that the drain sets after copying — the engine waits on it
        # before reusing the slot, making buffer corruption impossible
        # regardless of drain latency.
        _NUM_PINNED = max(2, min(3, self.config.dma_buffers))
        self._pinned_bufs: list[torch.Tensor | None] = [None] * _NUM_PINNED
        self._pinned_shapes: list[tuple] = [()] * _NUM_PINNED
        self._pinned_idx: int = 0
        self._num_pinned: int = _NUM_PINNED
        # One release event per buffer slot — set when drain copies out
        self._pinned_released: list[threading.Event] = [threading.Event() for _ in range(_NUM_PINNED)]
        for ev in self._pinned_released:
            ev.set()  # all slots start as "available"

        # Refiner scale as a device tensor so torch.compile / CUDA graphs
        # see it as data rather than a traced Python constant.  Must NOT be
        # a plain float — that would bake the first value into the captured
        # graph and silently ignore changes on subsequent frames.
        self._refiner_scale_t: torch.Tensor | None = None  # initialised after model load
        self._refiner_hook_handle = None

        # Engine-level optimization: always set explicitly so a previous
        # engine's values don't leak into this one (process-global state).
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = not self.config.disable_cudnn_benchmark
            if self.config.disable_cudnn_benchmark:
                logger.info("[Optimized] cuDNN benchmark disabled (saves 2-5 GB workspace).")

        if self.config.high_matmul_precision:
            torch.set_float32_matmul_precision("high")
            logger.info("[Optimized] TF32 matmul precision enabled.")
        else:
            torch.set_float32_matmul_precision("highest")

        self.model = self._load_model()

        # Normalization constants on device (used by process_raw and process_prepared GPU paths)
        self._mean_t = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        # Initialise refiner scale tensor on the model device so it lives
        # inside the CUDA graph's address space and can be updated via .fill_().
        self._refiner_scale_t = torch.ones(1, device=self.device, dtype=self.config.model_dtype)

        # Register persistent refiner scale hook (must be before torch.compile).
        # The hook always multiplies — no Python conditional that would create
        # a guard and cause graph breaks or bake one branch into a CUDA graph.
        if self.model.refiner is not None:
            scale_tensor = self._refiner_scale_t  # capture tensor, not 'self'

            def _scale_hook(module: torch.nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
                return output * scale_tensor

            self._refiner_hook_handle = self.model.refiner.register_forward_hook(_scale_hook)

        # Model compilation: TensorRT or torch.compile.
        # Applied at instance level (not class decorator) to avoid conflict with
        # timm's FX-based feature extraction.  Must be after hook registration so
        # hooks are baked into the compiled graph instead of causing graph breaks.
        # Based on work by Marcel Lieb (https://github.com/MarcelLieb/CorridorKey)
        self._use_trt = False
        self._trt_model: nn.Module | None = None

        if sys.platform in ("linux", "win32") and self.device.type == "cuda":
            if self.config.tensorrt:
                self._use_trt = self._try_tensorrt_compile()

            if not self._use_trt and self.config.compile_mode not in ("none", ""):
                # Clear any stale dynamo state left over from previous engine
                # loads or code changes.  Without this, a second engine
                # instantiation in the same process can encounter "FX to
                # symbolically trace a dynamo-optimized function" errors.
                torch._dynamo.reset()

                # Enable FX graph cache so graph tracing + inductor lowering
                # (the single-threaded phases) are skipped on subsequent runs.
                # Only the first run pays the full compilation cost.
                import torch._inductor.config as _inductor_cfg

                _inductor_cfg.fx_graph_cache = True

                # CUDA-graph compile modes require fixed memory layout and
                # cannot tolerate graph breaks from empty_cache(), dynamic
                # tile loops, or dict caches.  Force-disable incompatible opts.
                _uses_cuda_graphs = self.config.compile_mode in (
                    "reduce-overhead", "max-autotune",
                )

                # torch.cuda.empty_cache() causes graph breaks under
                # torch.compile.  Force-disable cache clearing on both the
                # engine config AND the model's config so _maybe_clear_cache()
                # and effective_cache_clearing are no-ops everywhere.
                if self.config.cache_clearing:
                    self.config = dataclasses.replace(self.config, cache_clearing=False)
                    if hasattr(self.model, "config") and self.model.config.cache_clearing:
                        self.model.config = dataclasses.replace(self.model.config, cache_clearing=False)
                    logger.info("[Compile] Cache clearing force-disabled (incompatible with torch.compile).")

                # Tiled refiner uses Python for-loops, dict caches, and
                # dynamic torch.zeros — all incompatible with CUDA graphs.
                # Swap to the non-tiled CNNRefiner for these modes.
                if _uses_cuda_graphs and self.config.tiled_refiner:
                    self.config = dataclasses.replace(self.config, tiled_refiner=False)
                    if hasattr(self.model, "config"):
                        self.model.config = dataclasses.replace(self.model.config, tiled_refiner=False)
                    # Replace TiledCNNRefiner with plain CNNRefinerModule
                    if hasattr(self.model, "refiner") and self.model.refiner is not None:
                        from .core.model_transformer import CNNRefinerModule
                        from .core.optimized_model import TiledCNNRefiner
                        if isinstance(self.model.refiner, TiledCNNRefiner):
                            plain = CNNRefinerModule(
                                in_channels=self.model.refiner.stem[0].in_channels,
                                hidden_channels=self.model.refiner.stem[0].out_channels,
                                out_channels=self.model.refiner.final.out_channels,
                            )
                            plain.load_state_dict(self.model.refiner.state_dict(), strict=False)
                            plain = plain.to(device=self.device, dtype=self.config.model_dtype)
                            plain.eval()
                            self.model.refiner = plain
                            # Re-register the scale hook on the new refiner
                            if self._refiner_hook_handle is not None:
                                self._refiner_hook_handle.remove()
                            scale_tensor = self._refiner_scale_t
                            def _scale_hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
                                return output * scale_tensor
                            self._refiner_hook_handle = self.model.refiner.register_forward_hook(_scale_hook)
                    logger.info("[Compile] Tiled refiner replaced with plain refiner (incompatible with CUDA graphs).")

                # Only disable dynamo on the encoder if timm uses FX-based
                # FeatureGraphNet (timm <1.0). Newer timm uses FeatureGetterNet
                # which is dynamo-compatible and should NOT be disabled.
                if hasattr(self.model, "encoder"):
                    encoder_type = type(self.model.encoder).__name__
                    if encoder_type == "FeatureGraphNet":
                        self.model.encoder.forward = torch._dynamo.disable(self.model.encoder.forward)
                        logger.info("[Compile] Encoder is FX-traced (%s) — dynamo skip applied.", encoder_type)
                    else:
                        logger.debug("[Compile] Encoder is %s — no dynamo skip needed.", encoder_type)

                self.model = torch.compile(self.model, mode=self.config.compile_mode)
                mode_label = self.config.compile_mode
                logger.info("[Optimized] torch.compile enabled (mode=%s, FX cache).", mode_label)

        # CUDA graph capture state (populated by capture_cuda_graph())
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_input: torch.Tensor | None = None
        self._graph_output: dict[str, torch.Tensor] | None = None

        # Create copy stream after model is loaded (avoids interfering with compile)
        if self.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self.device)
            # Pre-allocated CUDA events for timing (avoids per-frame cudaEventCreate)
            self._ev_timing = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        # Freeze batch size at init time (VRAM is most stable right after
        # model load).  Must not change at runtime or torch.compile will
        # retrace for the new shape.
        self._max_batch_size = self._compute_max_batch_size()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Instantiate and return the model (not yet loaded with weights).

        The returned model will be moved to ``self.device``, set to
        ``eval()`` mode, and loaded with the checkpoint by
        :meth:`_load_model`.
        """
        ...

    def _report_load_results(self, missing: list[str], unexpected: list[str]) -> None:
        """Log missing / unexpected state-dict keys after loading.

        The default implementation warns about any missing or unexpected
        keys.  ``OptimizedCorridorKeyEngine`` overrides this to handle
        expected LTRM keys gracefully.
        """
        if missing:
            logger.warning("[Warning] Missing keys: %s", missing)
        if unexpected:
            logger.warning("[Warning] Unexpected keys: %s", unexpected)

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def capture_cuda_graph(self) -> None:
        """Capture the model forward pass as a CUDA graph.

        Must be called after torch.compile warmup so that the compiled
        kernels are stable.  The captured graph replays the exact same
        kernel sequence on every frame, eliminating CPU-side kernel
        launch overhead.

        Requires ``config.cuda_graphs=True`` and a CUDA device.
        """
        if self.device.type != "cuda":
            logger.info("[CUDA Graph] Skipped — not a CUDA device.")
            return

        logger.info("[CUDA Graph] Running warmup passes before capture...")

        static_input = torch.zeros(
            1,
            4,
            self.img_size,
            self.img_size,
            device=self.device,
            dtype=self.config.model_dtype,
        )

        # Warmup: run 3 eager passes to stabilize cuDNN/cublas workspace
        for _ in range(3):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.config.effective_mixed_precision,
            ):
                _ = self.model(static_input)
        torch.cuda.synchronize(self.device)

        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._graph_input = static_input

        with (
            torch.cuda.graph(self._cuda_graph, stream=torch.cuda.current_stream(self.device)),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.config.effective_mixed_precision,
            ),
        ):
            self._graph_output = self.model(static_input)

        logger.info("[CUDA Graph] Captured successfully.")

    def _run_forward(self, inp_t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model forward pass, using CUDA graph replay if captured.

        When a CUDA graph is captured, replays it with ``inp_t`` copied
        into the static input buffer.  Otherwise, runs the model normally
        under autocast.
        """
        if self._cuda_graph is not None:
            self._graph_input.copy_(inp_t)
            self._cuda_graph.replay()
            return self._graph_output

        if self._use_trt:
            alpha, fg = self._trt_model(inp_t)
            return {"alpha": alpha, "fg": fg}

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.config.effective_mixed_precision,
        ):
            return self.model(inp_t)

    # ------------------------------------------------------------------
    # TensorRT compilation
    # ------------------------------------------------------------------

    def _try_tensorrt_compile(self) -> bool:
        """Attempt to compile the model with TensorRT.

        Returns ``True`` on success, ``False`` on failure (missing package,
        unsupported ops, etc.).  On failure, the caller falls back to
        ``torch.compile``.
        """
        try:
            import torch_tensorrt
        except ImportError:
            logger.warning("[TensorRT] torch-tensorrt not installed — falling back to torch.compile.")
            return False

        try:
            logger.info("[TensorRT] Compiling model (this may take several minutes on first run)...")
            wrapper = _GreenFormerTRT(self.model)

            self._trt_model = torch_tensorrt.compile(
                wrapper,
                inputs=[
                    torch_tensorrt.Input(
                        shape=[1, 4, self.img_size, self.img_size],
                        dtype=torch.float16,
                    )
                ],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,  # 1 GB workspace
                truncate_long_and_double=True,
            )
            logger.info("[TensorRT] Compilation successful.")
            return True
        except Exception as e:
            logger.warning("[TensorRT] Compilation failed: %s", e)
            logger.warning("[TensorRT] Falling back to torch.compile.")
            return False

    # ------------------------------------------------------------------
    # Checkpoint loading (shared)
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        """Load the checkpoint into the model returned by :meth:`_create_model`."""
        logger.info("Loading CorridorKey from %s...", self.checkpoint_path)

        model = self._create_model()
        model = model.to(self.config.model_dtype).to(self.device)
        model.eval()

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix compiled-model prefix & handle PosEmbed mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            # PosEmbed interpolation
            if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
                logger.info("Resizing %s from %s to %s", k, v.shape, model_state[k].shape)
                N_src = v.shape[1]
                C = v.shape[2]
                grid_src = int(math.sqrt(N_src))
                N_dst = model_state[k].shape[1]
                grid_dst = int(math.sqrt(N_dst))

                v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
                v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        self._report_load_results(missing, unexpected)

        # Report VRAM after loading
        if self.config.model_precision != "float32":
            logger.info("[Optimized] Model weights stored in %s.", self.config.model_precision)
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            logger.info("Model loaded. GPU memory: %.2f GB", allocated)

        return model

    # ------------------------------------------------------------------
    # GPU-side helpers
    # ------------------------------------------------------------------

    def _get_checkerboard_linear_gpu(self, w: int, h: int) -> torch.Tensor:
        """Return a cached checkerboard tensor [H, W, 3] on device in linear space.

        Only one resolution is kept in memory at a time.  If the requested
        (w, h) differs from the cached entry the old GPU tensor is evicted
        before the new one is allocated, preventing unbounded VRAM growth in
        mixed-resolution workflows.
        """
        key = (w, h)
        if self._checker_cache_key != key:
            self._checker_cache_val = None  # release old GPU tensor immediately
            y_coords = torch.arange(h, device=self.device) // DEFAULT_CHECKER_SIZE
            x_coords = torch.arange(w, device=self.device) // DEFAULT_CHECKER_SIZE
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
            checker = ((x_grid + y_grid) % 2).float()
            # Map 0 -> DEFAULT_CHECKER_COLOR1, 1 -> DEFAULT_CHECKER_COLOR2 (sRGB), then convert to linear before caching
            bg_srgb = checker * (DEFAULT_CHECKER_COLOR2 - DEFAULT_CHECKER_COLOR1) + DEFAULT_CHECKER_COLOR1  # [H, W]
            bg_srgb_3 = bg_srgb.unsqueeze(-1).expand(-1, -1, 3)
            self._checker_cache_val = cu.srgb_to_linear(bg_srgb_3)
            self._checker_cache_key = key
        return self._checker_cache_val  # type: ignore[return-value]

    def _get_checkerboard_linear_cpu(self, w: int, h: int) -> np.ndarray:
        """Return a cached checkerboard array [H, W, 3] in linear space.

        Only one resolution is kept in memory at a time (see GPU counterpart).
        """
        key = (w, h)
        if self._checker_cache_cpu_key != key:
            bg_srgb = cu.create_checkerboard(w, h, checker_size=DEFAULT_CHECKER_SIZE, color1=DEFAULT_CHECKER_COLOR1, color2=DEFAULT_CHECKER_COLOR2)
            self._checker_cache_cpu_val = cu.srgb_to_linear(bg_srgb)
            self._checker_cache_cpu_key = key
        return self._checker_cache_cpu_val  # type: ignore[return-value]

    @staticmethod
    def _clean_matte_gpu(alpha: torch.Tensor, area_threshold: int, dilation: int, blur_size: int) -> torch.Tensor:
        """Fully GPU matte cleanup using morphological operations.

        Approximates connected-component removal by eroding small regions
        away, then dilating back.  Avoids the GPU→CPU→GPU roundtrip that
        ``cv2.connectedComponentsWithStats`` would require.

        The erosion radius is derived from ``area_threshold``: a circular
        spot of area A has radius sqrt(A/pi), so erosion by that radius
        eliminates it.
        """
        _device = alpha.device
        # alpha: [H, W, 1]
        a2d = alpha[:, :, 0]
        mask = (a2d > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Erode: kill spots smaller than area_threshold
        # A circle of area A has radius r = sqrt(A / pi)

        erode_r = max(1, int(math.sqrt(area_threshold / math.pi)))
        erode_k = erode_r * 2 + 1
        # Erosion = negative of max_pool on negated mask
        mask = -F.max_pool2d(-mask, erode_k, stride=1, padding=erode_r)

        # Dilate back to restore edges of large regions
        dilate_r = erode_r + (dilation if dilation > 0 else 0)
        repeats = dilate_r // 2
        for _ in range(repeats):
            mask = F.max_pool2d(mask, 5, stride=1, padding=2)

        # Blur for soft edges
        if blur_size > 0:
            k = int(blur_size * 2 + 1)
            mask = TF.gaussian_blur(mask, [k, k])

        safe = mask.squeeze(0).squeeze(0)  # [H, W]
        return (a2d * safe).unsqueeze(-1)  # [H, W, 1]

    @staticmethod
    def _despill_gpu(image: torch.Tensor, strength: float) -> torch.Tensor:
        """GPU despill — keeps data on device."""
        if strength <= 0.0:
            return image
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        limit = (r + b) / 2.0
        spill = torch.clamp(g - limit, min=0.0)
        g_new = g - spill
        r_new = r + spill * 0.5
        b_new = b + spill * 0.5
        despilled = torch.stack([r_new, g_new, b_new], dim=-1)
        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled

    # ------------------------------------------------------------------
    # Frame processing (shared -- THE single implementation)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = DEFAULT_DESPILL_STRENGTH,
        auto_despeckle: bool = True,
        despeckle_size: int = DEFAULT_DESPECKLE_SIZE,
    ) -> dict[str, np.ndarray]:
        """Process a single frame.

        Args:
            image: ``[H, W, 3]`` numpy array (0-1 float or 0-255 uint8).
                sRGB by default; linear if *input_is_linear* is True.
            mask_linear: ``[H, W]`` or ``[H, W, 1]`` numpy array (0-1).
                The coarse alpha-hint mask.
            refiner_scale: Multiplier for refiner output deltas.
            input_is_linear: If True, resizes in linear then converts to sRGB.
            fg_is_straight: If True, foreground is straight (unpremultiplied).
            despill_strength: Green-spill removal strength (0-1).
            auto_despeckle: Remove small disconnected alpha components.
            despeckle_size: Minimum pixel area to keep.

        Returns:
            Dictionary with keys ``"alpha"``, ``"fg"``, ``"comp"``,
            and ``"processed"``.
        """
        use_gpu_postprocess = self.device.type == "cuda" and self.config.gpu_postprocess

        # === 1. Input normalization ===
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # === 2. Resize to model size ===
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # === 3. Prepare tensor and normalize on GPU ===
        inp_np = np.concatenate([img_resized, mask_resized], axis=-1)  # [H, W, 4]
        # Transfer to GPU, then normalize there to avoid CPU float intermediates
        inp_cpu = torch.as_tensor(inp_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        inp_t = inp_cpu.to(self.device, non_blocking=True)

        # Normalize RGB channels on GPU, leave mask channel as-is
        # (this op depends on inp_t, so CUDA stream ordering guarantees
        # the H2D DMA completes before it runs; inp_cpu ref keeps numpy alive)
        inp_t[:, :3] = (inp_t[:, :3] - self._mean_t) / self._std_t
        del inp_cpu  # safe to release after first GPU op that depends on inp_t

        # === 4. Inference ===
        self._refiner_scale_t.fill_(refiner_scale)

        # Cast input to match model weights (e.g. fp16) when autocast is off.
        if inp_t.dtype != self.config.model_dtype:
            inp_t = inp_t.to(dtype=self.config.model_dtype)

        out = self._run_forward(inp_t)

        pred_alpha = out["alpha"]  # [1, 1, model_size, model_size]
        pred_fg = out["fg"]  # [1, 3, model_size, model_size]

        # === 5. Post-process ===
        if use_gpu_postprocess:
            result = self._postprocess_gpu(
                pred_alpha,
                pred_fg,
                h,
                w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )
        else:
            result = self._postprocess_cpu(
                pred_alpha,
                pred_fg,
                h,
                w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )

        return result

    @torch.inference_mode()
    def process_prepared(
        self,
        inp_hwc4: np.ndarray,
        orig_h: int,
        orig_w: int,
        refiner_scale: float = 1.0,
        fg_is_straight: bool = True,
        despill_strength: float = DEFAULT_DESPILL_STRENGTH,
        auto_despeckle: bool = True,
        despeckle_size: int = DEFAULT_DESPECKLE_SIZE,
        _timings: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Fast path: process a pre-prepared [H,W,4] model input.

        Skips all CPU preprocessing (resize, normalize, concat) — those are
        done in the reader process.  The only GIL-held work here is
        ``torch.as_tensor().to(device)`` which is a single memcpy.

        If ``_timings`` dict is passed, per-stage wall-clock times (seconds)
        are written into it: ``upload``, ``forward``, ``postprocess``.
        """
        import time as _time

        t0 = _time.perf_counter() if _timings is not None else 0

        # Single tensor creation + GPU upload (minimal GIL time)
        # Combine dtype + device into one .to() call to avoid intermediate copy
        inp_cpu = torch.as_tensor(inp_hwc4).permute(2, 0, 1).unsqueeze(0)
        inp_t = inp_cpu.to(device=self.device, dtype=self.config.model_dtype, non_blocking=True)
        # inp_cpu ref keeps numpy backing alive during async H2D DMA

        if _timings is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = _time.perf_counter()
            _timings["upload"] = t1 - t0
        else:
            t1 = 0

        # Inference (GIL released during CUDA kernels)
        self._refiner_scale_t.fill_(refiner_scale)
        out = self._run_forward(inp_t)

        if _timings is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = _time.perf_counter()
            _timings["forward"] = t2 - t1
        else:
            t2 = 0

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]

        # Post-process (GPU path keeps data on device, includes .cpu() transfer)
        if self.device.type == "cuda" and self.config.gpu_postprocess:
            result = self._postprocess_gpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )
        else:
            result = self._postprocess_cpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )

        if _timings is not None:
            _timings["postprocess"] = _time.perf_counter() - t2

        return result

    def _forward_raw(
        self,
        img_raw: np.ndarray,
        mask_raw: np.ndarray,
        input_is_linear: bool,
        refiner_scale: float,
        _timings: dict | None,
    ) -> tuple[dict[str, torch.Tensor], bool, tuple]:
        """Upload, preprocess on GPU, and run the forward pass.

        Returns ``(model_output, use_cuda, timing_state)`` where
        *timing_state* is an opaque tuple consumed by
        :meth:`_finish_raw_timings`.
        """
        use_cuda = self.device.type == "cuda"
        use_events = _timings is not None and use_cuda

        if use_events:
            ev_start, ev_preprocess, ev_forward, _ev_unused = self._ev_timing
            ev_start.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            import time as _time

            t0 = _time.perf_counter()

        # Upload raw decoded arrays to GPU
        if not img_raw.flags["C_CONTIGUOUS"]:
            img_raw = np.ascontiguousarray(img_raw)
        img_t = torch.from_numpy(img_raw).to(
            device=self.device,
            dtype=torch.float32,
        )
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)

        if not mask_raw.flags["C_CONTIGUOUS"]:
            mask_raw = np.ascontiguousarray(mask_raw)
        mask_t = torch.from_numpy(mask_raw).to(
            device=self.device,
            dtype=torch.float32,
        )
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)

        # Resize on GPU
        img_t = F.interpolate(img_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mask_t = F.interpolate(mask_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        if input_is_linear:
            img_t = cu.linear_to_srgb(img_t)

        img_t = (img_t - self._mean_t) / self._std_t

        inp_t = torch.cat([img_t, mask_t], dim=1).contiguous()
        inp_t = inp_t.to(dtype=self.config.model_dtype)

        if use_events:
            ev_preprocess.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            t1 = _time.perf_counter()

        # Forward pass
        self._refiner_scale_t.fill_(refiner_scale)
        out = self._run_forward(inp_t)

        if use_events:
            ev_forward.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            t2 = _time.perf_counter()

        # Build timing state for _finish_raw_timings
        import time as _time_mod

        t_post0 = _time_mod.perf_counter() if _timings is not None else 0

        if use_events:
            ts = ("events", ev_start, ev_preprocess, ev_forward, t_post0)
        elif _timings is not None:
            ts = ("perf", t0, t1, t2, t_post0)
        else:
            ts = None

        return out, use_cuda, ts

    @staticmethod
    def _finish_raw_timings(_timings: dict | None, ts: tuple | None) -> None:
        """Write timing entries from the state returned by _forward_raw."""
        if _timings is None or ts is None:
            return
        import time as _time_mod

        if ts[0] == "events":
            _, ev_start, ev_preprocess, ev_forward, t_post0 = ts
            _timings["preprocess"] = ev_start.elapsed_time(ev_preprocess) / 1000.0
            _timings["forward"] = ev_preprocess.elapsed_time(ev_forward) / 1000.0
            _timings["postprocess"] = _time_mod.perf_counter() - t_post0
        else:
            _, t0, t1, t2, t_post0 = ts
            _timings["preprocess"] = t1 - t0
            _timings["forward"] = t2 - t1
            _timings["postprocess"] = _time_mod.perf_counter() - t_post0

    @torch.inference_mode()
    def process_raw(
        self,
        img_raw: np.ndarray,
        mask_raw: np.ndarray,
        orig_h: int,
        orig_w: int,
        input_is_linear: bool = False,
        refiner_scale: float = 1.0,
        fg_is_straight: bool = True,
        despill_strength: float = DEFAULT_DESPILL_STRENGTH,
        auto_despeckle: bool = True,
        despeckle_size: int = DEFAULT_DESPECKLE_SIZE,
        _timings: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """GPU-accelerated path: upload raw decoded frames and preprocess on device.

        Moves resize, color space conversion, and normalization to GPU,
        reducing CPU→GPU transfer size and eliminating CPU-bound preprocessing.
        """
        out, use_cuda, ts = self._forward_raw(
            img_raw,
            mask_raw,
            input_is_linear,
            refiner_scale,
            _timings,
        )

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]

        if use_cuda and self.config.gpu_postprocess:
            result = self._postprocess_gpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )
        else:
            result = self._postprocess_cpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )

        self._finish_raw_timings(_timings, ts)
        return result

    @torch.inference_mode()
    def process_raw_deferred(
        self,
        img_raw: np.ndarray,
        mask_raw: np.ndarray,
        orig_h: int,
        orig_w: int,
        input_is_linear: bool = False,
        refiner_scale: float = 1.0,
        fg_is_straight: bool = True,
        despill_strength: float = DEFAULT_DESPILL_STRENGTH,
        auto_despeckle: bool = True,
        despeckle_size: int = DEFAULT_DESPECKLE_SIZE,
        _timings: dict | None = None,
    ) -> PendingTransfer:
        """Like :meth:`process_raw` but returns a :class:`PendingTransfer`.

        The GPU→CPU DMA is started but not waited on.  Call
        ``pending.resolve()`` to block until the transfer completes and
        get the numpy dict.  This allows overlapping the DMA with the
        next frame's preprocess + forward pass.
        """
        # Skip CUDA event timing — reading elapsed times requires
        # synchronizing the forward event, which blocks the CPU thread
        # and defeats the purpose of deferring.  The inference worker's
        # profiler captures wall-clock timings independently.
        out, use_cuda, ts = self._forward_raw(
            img_raw,
            mask_raw,
            input_is_linear,
            refiner_scale,
            None,
        )

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]

        if use_cuda and self.config.gpu_postprocess:
            pending = self._postprocess_gpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
                sync=False,
            )
        else:
            result = self._postprocess_cpu(
                pred_alpha,
                pred_fg,
                orig_h,
                orig_w,
                auto_despeckle,
                despeckle_size,
                despill_strength,
                fg_is_straight,
            )
            pending = PendingTransfer(None, None, result, None)

        self._finish_raw_timings(_timings, ts)
        return pending

    @torch.inference_mode()
    def process_raw_batch(
        self,
        frames: list[tuple[np.ndarray, np.ndarray, int, int]],
        input_is_linear: bool = False,
        refiner_scale: float = 1.0,
        fg_is_straight: bool = True,
        despill_strength: float = DEFAULT_DESPILL_STRENGTH,
        auto_despeckle: bool = True,
        despeckle_size: int = DEFAULT_DESPECKLE_SIZE,
        _timings: dict | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Batched GPU inference: process multiple frames in one forward pass.

        Parameters
        ----------
        frames : list of (img_raw, mask_raw, orig_h, orig_w)
            Each element is a raw decoded frame pair at original resolution.
        _timings : dict | None
            If provided, per-stage times are written (amortized over batch).

        Returns
        -------
        list of dicts, one per frame, each with "alpha", "fg", "comp", "processed".
        """
        import time as _time_mod

        B = len(frames)
        if B == 0:
            return []

        use_cuda = self.device.type == "cuda"
        use_events = _timings is not None and use_cuda

        if use_events:
            ev_start, ev_preprocess, ev_forward, _ = self._ev_timing
            ev_start.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            t0 = _time_mod.perf_counter()

        # === Preprocess: upload and resize each frame, then stack ===
        # All frames go to GPU individually (each may have different orig
        # resolution), then get resized to model size and stacked.
        img_batch = []
        _mask_batch = []
        orig_sizes = []
        for img_raw, mask_raw, orig_h, orig_w in frames:
            if not img_raw.flags["C_CONTIGUOUS"]:
                img_raw = np.ascontiguousarray(img_raw)
            img_t = torch.from_numpy(img_raw).to(device=self.device, dtype=torch.float32)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)

            if not mask_raw.flags["C_CONTIGUOUS"]:
                mask_raw = np.ascontiguousarray(mask_raw)
            mask_t = torch.from_numpy(mask_raw).to(device=self.device, dtype=torch.float32)
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)

            img_t = F.interpolate(img_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            mask_t = F.interpolate(mask_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

            if input_is_linear:
                img_t = cu.linear_to_srgb(img_t)

            img_t = (img_t - self._mean_t) / self._std_t

            img_batch.append(torch.cat([img_t, mask_t], dim=1))
            orig_sizes.append((orig_h, orig_w))

        # Single stack → one contiguous [B, 4, H, W] tensor for the model
        inp_t = torch.cat(img_batch, dim=0).contiguous()

        # Pad to fixed batch size so torch.compile only ever sees one shape
        # (zero recompilation).  Padding frames are zeros — harmless through
        # the model, and sliced off before postprocess.
        target_B = self.max_batch_size()
        if target_B > B:
            padding = torch.zeros(
                target_B - B,
                4,
                self.img_size,
                self.img_size,
                device=self.device,
                dtype=inp_t.dtype,
            )
            inp_t = torch.cat([inp_t, padding], dim=0)

        inp_t = inp_t.to(dtype=self.config.model_dtype)

        if use_events:
            ev_preprocess.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            t1 = _time_mod.perf_counter()

        # === Forward: one pass for the entire (padded) batch ===
        self._refiner_scale_t.fill_(refiner_scale)
        out = self._run_forward(inp_t)

        if use_events:
            ev_forward.record(torch.cuda.current_stream(self.device))
        elif _timings is not None:
            t2 = _time_mod.perf_counter()

        t_post0 = _time_mod.perf_counter() if _timings is not None else 0

        # === Postprocess: slice to real frames, process individually ===
        # Discard padding frames, then resize each independently (frames
        # may have different original resolutions).
        pred_alpha = out["alpha"][:B]  # [B, 1, model_h, model_w]
        pred_fg = out["fg"][:B]  # [B, 3, model_h, model_w]

        results = []
        if use_cuda:
            for i in range(B):
                result = self._postprocess_gpu(
                    pred_alpha[i : i + 1],
                    pred_fg[i : i + 1],
                    orig_sizes[i][0],
                    orig_sizes[i][1],
                    auto_despeckle,
                    despeckle_size,
                    despill_strength,
                    fg_is_straight,
                )
                results.append(result)
        else:
            for i in range(B):
                result = self._postprocess_cpu(
                    pred_alpha[i : i + 1],
                    pred_fg[i : i + 1],
                    orig_sizes[i][0],
                    orig_sizes[i][1],
                    auto_despeckle,
                    despeckle_size,
                    despill_strength,
                    fg_is_straight,
                )
                results.append(result)

        # Extract timings (amortized over batch)
        if use_events:
            _timings["preprocess"] = ev_start.elapsed_time(ev_preprocess) / 1000.0
            _timings["forward"] = ev_preprocess.elapsed_time(ev_forward) / 1000.0
            _timings["postprocess"] = _time_mod.perf_counter() - t_post0
        elif _timings is not None:
            _timings["preprocess"] = t1 - t0
            _timings["forward"] = t2 - t1
            _timings["postprocess"] = _time_mod.perf_counter() - t2

        return results

    def max_batch_size(self) -> int:
        """Return the frozen batch size computed at init time."""
        return self._max_batch_size

    def set_max_batch_size(self, size: int) -> None:
        """Override the batch size (must be called before warmup/inference)."""
        self._max_batch_size = size

    def _compute_max_batch_size(self) -> int:
        """Estimate max batch size based on available GPU memory.

        Called once at init (right after model load, when VRAM is most
        stable) and cached.  Must not be recomputed at runtime or
        torch.compile will retrace for the new padded shape.

        Conservative: uses 2 GB per frame estimate (activations + attention
        buffers at 2048²) and 30% headroom.  Caps at 8 to avoid diminishing
        returns from memory bandwidth saturation.
        """
        if self.device.type != "cuda":
            return 1

        free, _total = torch.cuda.mem_get_info(self.device)
        per_frame_mb = 2048  # ~2 GB per frame at 2048² (activations + attention)
        available_mb = (free * 0.7) / (1024**2)
        batch = max(1, int(available_mb / per_frame_mb))
        return min(batch, 8)  # cap at 8

    def _postprocess_gpu(
        self,
        pred_alpha: torch.Tensor,
        pred_fg: torch.Tensor,
        h: int,
        w: int,
        auto_despeckle: bool,
        despeckle_size: int,
        despill_strength: float,
        fg_is_straight: bool,
        sync: bool = True,
    ) -> dict[str, np.ndarray] | PendingTransfer:
        """Post-process on GPU, transfer final results to CPU.

        When ``sync=True`` (default), blocks until transfer completes and
        returns numpy arrays.  When ``sync=False``, starts the DMA
        non-blocking and returns a :class:`PendingTransfer` — call
        ``.resolve()`` to get the numpy dict later.
        """
        # Resize on GPU using F.interpolate (much faster than cv2 at 4K)
        alpha_up = F.interpolate(pred_alpha.float(), size=(h, w), mode="bilinear", align_corners=False)
        fg_up = F.interpolate(pred_fg.float(), size=(h, w), mode="bilinear", align_corners=False)

        # Convert to HWC on GPU
        res_alpha = alpha_up[0].permute(1, 2, 0)  # [H, W, 1]
        res_fg = fg_up[0].permute(1, 2, 0)  # [H, W, 3]

        # A. Clean matte
        if auto_despeckle:
            processed_alpha = self._clean_matte_gpu(res_alpha, despeckle_size, dilation=DEFAULT_MATTE_DILATION, blur_size=DEFAULT_MATTE_BLUR)
        else:
            processed_alpha = res_alpha

        # B. Despill on GPU
        fg_despilled = self._despill_gpu(res_fg, despill_strength)

        # C. sRGB → linear on GPU
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)

        # D. Premultiply on GPU
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

        # E. Pack RGBA on GPU
        processed_rgba = torch.cat([fg_premul_lin, processed_alpha], dim=-1)

        # F. Composite (optional — skip if comp output is disabled)
        if self.config.comp_format != "none":
            if self.config.comp_checkerboard:
                bg_lin = self._get_checkerboard_linear_gpu(w, h)
                if fg_is_straight:
                    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
                else:
                    comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
                comp_srgb = cu.linear_to_srgb(comp_lin)  # [H, W, 3] opaque
            else:
                # Transparent RGBA: straight fg in sRGB + alpha
                fg_srgb = cu.linear_to_srgb(fg_despilled_lin)
                comp_srgb = torch.cat([fg_srgb, processed_alpha], dim=-1)  # [H, W, 4]
        else:
            comp_srgb = None

        # === Bulk transfer to CPU via copy stream ===
        #
        # Pack all outputs into one contiguous tensor, DMA to a pinned
        # buffer on the copy stream.  Layout varies by comp mode:
        #   Checkerboard: [H, W, 1+3+3+4] = [alpha, fg, comp_rgb, processed]
        #   Transparent:  [H, W, 1+3+4+4] = [alpha, fg, comp_rgba, processed]
        #   No comp:      [H, W, 1+3+3+4] = [alpha, fg, zeros, processed]
        if comp_srgb is not None:
            comp_channels = comp_srgb.shape[-1]
            bulk = torch.cat([res_alpha, res_fg, comp_srgb, processed_rgba], dim=-1)
        else:
            comp_channels = 3
            bulk = torch.cat([res_alpha, res_fg, torch.zeros_like(res_fg), processed_rgba], dim=-1)

        if self._copy_stream is not None:
            # Select pinned buffer and rotate index for next call
            idx = self._pinned_idx
            self._pinned_idx = (self._pinned_idx + 1) % self._num_pinned

            # Wait for drain to finish copying from this slot before reuse.
            # With triple buffering this almost never blocks (the slot was
            # used 3 frames ago), but guarantees correctness if drain lags.
            self._pinned_released[idx].wait()
            self._pinned_released[idx].clear()

            # Reallocate this buffer slot if shape changed
            if self._pinned_shapes[idx] != bulk.shape:
                self._pinned_bufs[idx] = torch.empty(bulk.shape, dtype=bulk.dtype, pin_memory=True)
                self._pinned_shapes[idx] = bulk.shape

            pinned = self._pinned_bufs[idx]

            # Wait for compute to finish, then async DMA on copy stream
            compute_event = torch.cuda.current_stream(self.device).record_event()
            self._copy_stream.wait_event(compute_event)
            with torch.cuda.stream(self._copy_stream):
                pinned.copy_(bulk, non_blocking=True)

            if not sync:
                # Record event right after this DMA so resolve() waits
                # only for THIS transfer, not later ones that may reuse
                # the same pinned buffer.
                dma_event = self._copy_stream.record_event()
                pt = PendingTransfer(dma_event, pinned, None, self._pinned_released[idx], bulk)
                pt._comp_channels = comp_channels
                return pt

            # Sync path: block until DMA completes, return numpy with .copy()
            self._copy_stream.synchronize()
            bulk_np = pinned.numpy()
            cc = comp_channels
            result = {
                "alpha": bulk_np[:, :, 0:1].copy(),
                "fg": bulk_np[:, :, 1:4].copy(),
                "comp": bulk_np[:, :, 4 : 4 + cc].copy(),
                "processed": bulk_np[:, :, 4 + cc : 4 + cc + 4].copy(),
            }
            self._pinned_released[idx].set()
            return result
        bulk_np = bulk.cpu().numpy()
        cc = comp_channels
        result = {
            "alpha": bulk_np[:, :, 0:1],
            "fg": bulk_np[:, :, 1:4],
            "comp": bulk_np[:, :, 4 : 4 + cc],
            "processed": bulk_np[:, :, 4 + cc : 4 + cc + 4],
        }
        if not sync:
            return PendingTransfer(None, None, result, None)
        return result

    def _postprocess_cpu(
        self,
        pred_alpha: torch.Tensor,
        pred_fg: torch.Tensor,
        h: int,
        w: int,
        auto_despeckle: bool,
        despeckle_size: int,
        despill_strength: float,
        fg_is_straight: bool,
    ) -> dict[str, np.ndarray]:
        """CPU fallback post-processing (for non-CUDA devices)."""
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=DEFAULT_MATTE_DILATION, blur_size=DEFAULT_MATTE_BLUR)
        else:
            processed_alpha = res_alpha

        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        if self.config.comp_format != "none":
            if self.config.comp_checkerboard:
                bg_lin = self._get_checkerboard_linear_cpu(w, h)
                if fg_is_straight:
                    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
                else:
                    comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
                comp_srgb = cu.linear_to_srgb(comp_lin)  # [H, W, 3] opaque
            else:
                # Transparent RGBA: straight fg in sRGB + alpha
                fg_srgb = cu.linear_to_srgb(fg_despilled_lin)
                comp_srgb = np.concatenate([fg_srgb, processed_alpha], axis=-1)  # [H, W, 4]
        else:
            comp_srgb = None

        result = {
            "alpha": res_alpha,
            "fg": res_fg,
            "processed": processed_rgba,
        }
        if comp_srgb is not None:
            result["comp"] = comp_srgb
        return result
