"""Original CorridorKey inference engine.

Uses :class:`GreenFormer` with no VRAM optimizations by default.
Optimizations can be selectively enabled by passing an
:class:`OptimizationConfig`.
"""

from __future__ import annotations

from .base_engine import _BaseCorridorKeyEngine
from .core.model_transformer import GreenFormer
from .optimization_config import OptimizationConfig


class CorridorKeyEngine(_BaseCorridorKeyEngine):
    """Standard inference engine.

    By default uses :meth:`OptimizationConfig.original` (no optimizations).
    Pass a custom ``optimization_config`` to enable individual optimizations
    while using the original ``GreenFormer`` architecture.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            use_refiner=use_refiner,
            optimization_config=optimization_config or OptimizationConfig.original(),
        )

    def _create_model(self) -> GreenFormer:
        return GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
            optimization_config=self.config,
        )
