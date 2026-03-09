"""Original CorridorKey inference engine.

Uses :class:`GreenFormer` with VRAM optimizations enabled by default
(flash attention, tiled refiner, cache clearing).  These are
mathematically lossless and reduce peak VRAM from ~22 GB to ~3 GB.
Pass ``OptimizationConfig.original()`` explicitly to disable them.
"""

from __future__ import annotations

from .base_engine import _BaseCorridorKeyEngine
from .core.model_transformer import GreenFormer
from .optimization_config import OptimizationConfig


class CorridorKeyEngine(_BaseCorridorKeyEngine):
    """Standard inference engine.

    Defaults to VRAM-optimized settings (flash attention, tiled refiner,
    cache clearing, cuDNN benchmark off).  Pass a custom
    ``optimization_config`` to override.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        if optimization_config is None:
            optimization_config = OptimizationConfig(
                flash_attention=True,
                tiled_refiner=True,
                disable_cudnn_benchmark=True,
                cache_clearing=True,
                mixed_precision=True,
                model_precision="float16",
                high_matmul_precision=True,
            )
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            use_refiner=use_refiner,
            optimization_config=optimization_config,
        )

    def _create_model(self) -> GreenFormer:
        return GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
            optimization_config=self.config,
        )
