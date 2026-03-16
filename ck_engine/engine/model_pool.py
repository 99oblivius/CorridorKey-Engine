"""Model lifecycle management with config-hash reuse.

The ModelPool caches loaded GPU models and reuses them across jobs when
the configuration hasn't changed.  It emits model events through the
EventBus so frontends can track load times and VRAM usage.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ck_engine.engine.event_bus import EventBus

logger = logging.getLogger(__name__)


class ModelPool:
    """Manages GPU model lifecycle with config-hash caching."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus
        self._lock = threading.Lock()

        # Inference engine cache
        self._engine: Any = None
        self._engine_hash: str = ""
        self._engine_device: str = ""
        self._engine_backend: str = ""
        self._engine_img_size: int = 0
        self._engine_precision: str = ""

        # Generator cache
        self._generator: Any = None
        self._generator_name: str = ""
        self._generator_device: str = ""

    def get_inference_engine(
        self,
        backend: str | None = None,
        device: str | None = None,
        img_size: int = 2048,
        optimization_config: Any = None,
    ) -> Any:
        """Get or create an inference engine, reusing if config matches."""
        from ck_engine.api.events import ModelLoaded, ModelLoading, ModelRecompiling

        with self._lock:
            config_hash = self._hash_config(backend, device, img_size, optimization_config)

            if self._engine is not None and self._engine_hash == config_hash:
                logger.info("Reusing cached inference engine (config unchanged)")
                return self._engine

            # Config changed — unload old engine
            if self._engine is not None:
                self._emit(ModelRecompiling(
                    reason="configuration changed",
                    backend=self._engine_backend,
                ))
                self._unload_engine()

            # Resolve device and backend
            from ck_engine.device import resolve_device
            resolved_device = resolve_device(device)

            from CorridorKeyModule.engine_factory import create_engine, resolve_backend
            resolved_backend = resolve_backend(backend)

            self._emit(ModelLoading(model="inference", device=resolved_device))
            t0 = time.monotonic()

            engine = create_engine(
                backend=resolved_backend,
                device=resolved_device,
                img_size=img_size,
                optimization_config=optimization_config,
            )

            load_time = time.monotonic() - t0
            vram = self._get_vram_mb()

            self._engine = engine
            self._engine_hash = config_hash
            self._engine_device = resolved_device
            self._engine_backend = resolved_backend
            self._engine_img_size = img_size
            self._engine_precision = getattr(optimization_config, "model_precision", "float32") if optimization_config else "float32"

            self._emit(ModelLoaded(
                model="inference",
                device=resolved_device,
                vram_mb=vram,
                load_seconds=load_time,
            ))

            return engine

    def get_generator(self, name: str, device: str | None = None) -> Any:
        """Get or create an alpha generator, reusing if name+device match."""
        from ck_engine.api.events import ModelLoaded, ModelLoading

        with self._lock:
            from ck_engine.device import resolve_device
            resolved_device = resolve_device(device)

            if (self._generator is not None
                    and self._generator_name == name
                    and self._generator_device == resolved_device):
                logger.info("Reusing cached %s generator", name)
                return self._generator

            # Unload old generator
            if self._generator is not None:
                self._unload_generator()

            self._emit(ModelLoading(model=name, device=resolved_device))
            t0 = time.monotonic()

            from ck_engine.generators import get_generator
            generator = get_generator(name, device=resolved_device)

            load_time = time.monotonic() - t0
            vram = self._get_vram_mb()

            self._generator = generator
            self._generator_name = name
            self._generator_device = resolved_device

            self._emit(ModelLoaded(
                model=name,
                device=resolved_device,
                vram_mb=vram,
                load_seconds=load_time,
            ))

            return generator

    def unload(self, which: str = "all") -> float:
        """Unload models and return approximate freed VRAM in MB."""
        with self._lock:
            freed = 0.0
            if which in ("all", "inference") and self._engine is not None:
                freed += self._unload_engine()
            if which in ("all", "generator") and self._generator is not None:
                freed += self._unload_generator()
            return freed

    def status(self) -> dict:
        """Return status of loaded models."""
        with self._lock:
            result: dict[str, Any] = {
                "inference_engine": None,
                "generator": None,
            }
            if self._engine is not None:
                result["inference_engine"] = {
                    "backend": self._engine_backend,
                    "device": self._engine_device,
                    "vram_mb": self._get_vram_mb(),
                    "config_hash": self._engine_hash,
                    "img_size": self._engine_img_size,
                    "precision": self._engine_precision,
                }
            if self._generator is not None:
                result["generator"] = {
                    "backend": self._generator_name,
                    "device": self._generator_device,
                    "vram_mb": 0.0,
                }
            return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unload_engine(self) -> float:
        """Unload inference engine. Caller must hold _lock."""
        from ck_engine.api.events import ModelUnloaded

        if self._engine is None:
            return 0.0
        vram_before = self._get_vram_mb()

        engine = self._engine
        self._engine = None

        # --- Tear down CUDA graphs before anything else ---
        for attr in ("_cuda_graph", "_graph_input", "_graph_output"):
            try:
                setattr(engine, attr, None)
            except Exception:
                pass

        # --- Remove refiner hook to break reference cycle ---
        try:
            if hasattr(engine, "_refiner_hook_handle") and engine._refiner_hook_handle is not None:
                engine._refiner_hook_handle.remove()
                engine._refiner_hook_handle = None
        except Exception:
            pass

        # --- Move the raw model to CPU (bypass torch.compile wrapper) ---
        try:
            model = getattr(engine, "model", None)
            if model is not None:
                raw = getattr(model, "_orig_mod", model)
                raw.cpu()
        except Exception:
            pass

        # --- Reset torch.compile / dynamo state AND inductor FX cache ---
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:
            pass
        try:
            import torch._inductor.config as _ind_cfg
            # Invalidate the in-process FX graph cache so stale compiled
            # kernels (referencing freed CUDA addresses) are not reused
            # when a new engine is loaded.
            if hasattr(_ind_cfg, "fx_graph_cache"):
                _ind_cfg.fx_graph_cache = False
        except Exception:
            pass

        # --- Synchronize CUDA before GC ---
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        del engine
        gc.collect()
        self._clear_device_cache()

        # --- Restore global defaults that engine __init__ may have changed ---
        try:
            import torch
            torch.backends.cudnn.benchmark = True  # PyTorch default
            torch.set_float32_matmul_precision("highest")  # PyTorch default
        except Exception:
            pass

        freed = max(0.0, vram_before - self._get_vram_mb())
        self._emit(ModelUnloaded(model="inference", freed_mb=freed))
        self._engine_hash = ""
        return freed

    def _unload_generator(self) -> float:
        """Unload generator. Caller must hold _lock."""
        from ck_engine.api.events import ModelUnloaded

        if self._generator is None:
            return 0.0
        name = self._generator_name
        vram_before = self._get_vram_mb()
        self._generator = None
        gc.collect()
        self._clear_device_cache()
        freed = max(0.0, vram_before - self._get_vram_mb())
        self._emit(ModelUnloaded(model=name, freed_mb=freed))
        self._generator_name = ""
        self._generator_device = ""
        return freed

    def _emit(self, event: Any) -> None:
        """Emit an event if event_bus is available."""
        if self._event_bus is not None:
            self._event_bus.emit(event)

    @staticmethod
    def _hash_config(backend: str | None, device: str | None, img_size: int, opt_config: Any) -> str:
        """Hash the config inputs to determine cache reuse."""
        parts: dict[str, Any] = {
            "backend": backend or "auto",
            "device": device or "auto",
            "img_size": img_size,
        }
        if opt_config is not None:
            # Serialize all OptimizationConfig fields
            import dataclasses
            if dataclasses.is_dataclass(opt_config):
                parts["optimization"] = dataclasses.asdict(opt_config)
            else:
                parts["optimization"] = str(opt_config)
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _get_vram_mb() -> float:
        """Get current VRAM usage in MB. Returns 0 if not on CUDA."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _clear_device_cache() -> None:
        """Clear GPU cache."""
        try:
            from ck_engine.device import clear_device_cache
            clear_device_cache("auto")
        except Exception:
            pass
