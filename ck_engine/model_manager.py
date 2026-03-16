"""ModelManager — owns all heavy model lifecycle for CorridorKey.

Enforces the one-model-at-a-time VRAM residency policy: before loading a
new model type, the current model is moved to CPU and VRAM is freed via
backend.device.clear_device_cache(). Prevents OOM on 24 GB cards.

Intended usage::

    mm = ModelManager()
    mm.detect_device()

    with mm.gpu_lock:
        engine = mm.get_engine()

    mm.unload_engines()
"""

from __future__ import annotations

# DEPRECATED: This module is superseded by backend.engine.model_pool.ModelPool.
# It remains for backwards compatibility and will be removed in a future release.
import warnings as _warnings
_warnings.warn(
    "ck_engine.model_manager is deprecated, use backend.engine.model_pool instead",
    DeprecationWarning,
    stacklevel=2,
)

import gc
import logging
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CorridorKeyModule.base_engine import _BaseCorridorKeyEngine
    from CorridorKeyModule.engine_factory import _MLXEngineAdapter
    from ck_engine.generators.gvm.wrapper import GVMProcessor
    from ck_engine.generators.videomama.pipeline import VideoInferencePipeline

logger = logging.getLogger(__name__)


class _ActiveModel(Enum):
    """Tracks which heavy model is currently loaded in VRAM."""

    NONE = "none"
    INFERENCE = "inference"
    GVM = "gvm"
    VIDEOMAMA = "videomama"


class ModelManager:
    """Owns model references and serialises all GPU lifecycle operations.

    Only one heavy model is resident in VRAM at any time.  Callers must
    acquire ``gpu_lock`` before calling ``get_engine``, ``get_gvm``, or
    ``get_videomama_pipeline`` and keep it held for the duration of the
    forward pass so that concurrent threads cannot trigger an unload mid-
    inference.
    """

    def __init__(self) -> None:
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model: _ActiveModel = _ActiveModel.NONE
        self._device: str = "cpu"
        # GPU mutex — serialises ALL model operations (Codex: thread safety)
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def gpu_lock(self) -> threading.Lock:
        """Lock that must be held for any model load or forward-pass call."""
        return self._gpu_lock

    @property
    def device(self) -> str:
        """Current compute device string, e.g. ``"cuda"`` or ``"cpu"``."""
        return self._device

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def detect_device(self) -> str:
        """Detect the best available compute device via backend.device."""
        from .device import resolve_device

        self._device = resolve_device()
        logger.info(f"Compute device: {self._device}")
        return self._device

    def get_vram_info(self) -> dict[str, float]:
        """Return GPU VRAM stats in GB.  Empty dict when CUDA unavailable."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_mem
            reserved = torch.cuda.memory_reserved(0)
            return {
                "total": total_bytes / (1024**3),
                "reserved": reserved / (1024**3),
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "free": (total_bytes - reserved) / (1024**3),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug(f"VRAM query failed: {e}")
            return {}

    @staticmethod
    def _vram_allocated_mb() -> float:
        """Return VRAM currently allocated in MB, or 0 if unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024**2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe_offload(obj: object) -> None:
        """Move a model's GPU tensors to CPU before dropping the reference.

        Handles diffusers pipelines (.to('cpu')), plain nn.Modules (.cpu()),
        and objects with an explicit unload() method.
        """
        if obj is None:
            return
        logger.debug(f"Offloading model: {type(obj).__name__}")
        try:
            if hasattr(obj, "unload"):
                obj.unload()
            elif hasattr(obj, "to"):
                obj.to("cpu")
            elif hasattr(obj, "cpu"):
                obj.cpu()
        except Exception as e:
            logger.debug(f"Model offload warning: {e}")

    # ------------------------------------------------------------------
    # Model residency
    # ------------------------------------------------------------------

    def _ensure_model(self, needed: _ActiveModel) -> None:
        """Model residency manager — unload current model when switching types.

        Only ONE heavy model stays in VRAM at a time.  Before loading a
        different model, the previous is moved to CPU and dereferenced so
        the allocator can reclaim VRAM.
        """
        if self._active_model == needed:
            return

        if self._active_model != _ActiveModel.NONE:
            vram_before_mb = self._vram_allocated_mb()
            logger.info(
                f"Unloading {self._active_model.value} model for {needed.value} (VRAM before: {vram_before_mb:.0f}MB)"
            )

            if self._active_model == _ActiveModel.INFERENCE:
                self._safe_offload(self._engine)
                self._engine = None
            elif self._active_model == _ActiveModel.GVM:
                self._safe_offload(self._gvm_processor)
                self._gvm_processor = None
            elif self._active_model == _ActiveModel.VIDEOMAMA:
                self._safe_offload(self._videomama_pipeline)
                self._videomama_pipeline = None

            gc.collect()

            from .device import clear_device_cache

            clear_device_cache(self._device)

            vram_after_mb = self._vram_allocated_mb()
            freed = vram_before_mb - vram_after_mb
            logger.info(f"VRAM after unload: {vram_after_mb:.0f}MB (freed {freed:.0f}MB)")

        self._active_model = needed

    # ------------------------------------------------------------------
    # Lazy model accessors (public API)
    # ------------------------------------------------------------------

    def get_engine(self) -> _BaseCorridorKeyEngine | _MLXEngineAdapter:
        """Lazy-load the CorridorKey inference engine via the backend factory."""
        self._ensure_model(_ActiveModel.INFERENCE)

        if self._engine is not None:
            return self._engine

        from CorridorKeyModule.engine_factory import create_engine
        from .config import DEFAULT_IMG_SIZE

        t0 = time.monotonic()
        self._engine = create_engine(
            backend="auto",
            device=self._device,
            img_size=DEFAULT_IMG_SIZE,
        )
        logger.info(f"Engine loaded in {time.monotonic() - t0:.1f}s")
        return self._engine

    def get_gvm(self) -> GVMProcessor:
        """Lazy-load the GVM processor."""
        self._ensure_model(_ActiveModel.GVM)

        if self._gvm_processor is not None:
            return self._gvm_processor

        from ck_engine.generators.gvm import GVMProcessor

        logger.info("Loading GVM processor...")
        t0 = time.monotonic()
        self._gvm_processor = GVMProcessor(device=self._device)
        logger.info(f"GVM loaded in {time.monotonic() - t0:.1f}s")
        return self._gvm_processor

    def get_videomama_pipeline(self) -> VideoInferencePipeline:
        """Lazy-load the VideoMaMa inference pipeline."""
        self._ensure_model(_ActiveModel.VIDEOMAMA)

        if self._videomama_pipeline is not None:
            return self._videomama_pipeline

        from ck_engine.generators.videomama.inference import load_videomama_model

        logger.info("Loading VideoMaMa pipeline...")
        t0 = time.monotonic()
        self._videomama_pipeline = load_videomama_model(device=self._device)
        logger.info(f"VideoMaMa loaded in {time.monotonic() - t0:.1f}s")
        return self._videomama_pipeline

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload_engines(self) -> None:
        """Free GPU memory by unloading all engines."""
        self._safe_offload(self._engine)
        self._safe_offload(self._gvm_processor)
        self._safe_offload(self._videomama_pipeline)
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model = _ActiveModel.NONE

        from .device import clear_device_cache

        clear_device_cache(self._device)
        logger.info("All engines unloaded, VRAM freed")

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is currently resident in VRAM."""
        return self._active_model == _ActiveModel.INFERENCE and self._engine is not None
