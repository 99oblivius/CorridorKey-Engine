"""VRAM-optimized inference engine.

Drop-in replacement for :class:`CorridorKeyEngine`.  Uses
:class:`OptimizedGreenFormer` with tiled CNN refiner, FlashAttention
patching, and memory hygiene to reduce peak VRAM from ~22.7 GB to under
8 GB.  Optional hint-based token routing can be enabled for further
savings (requires fine-tuning).
"""

from __future__ import annotations

import torch.nn.functional as F

from .base_engine import _BaseCorridorKeyEngine
from .core.optimized_model import OptimizedGreenFormer
from .optimization_config import OptimizationConfig


class OptimizedCorridorKeyEngine(_BaseCorridorKeyEngine):
    """Inference engine with VRAM optimizations.

    API-compatible with :class:`CorridorKeyEngine`.  Same constructor
    signature (plus extra tuning knobs), identical ``process_frame()``
    output contract.

    The constructor accepts either an ``optimization_config`` or the
    legacy per-parameter API for backward compatibility.  When both are
    provided, ``optimization_config`` takes precedence.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        # Legacy per-param API (backward compat)
        use_token_routing: bool = False,
        edge_threshold_low: float = 0.02,
        edge_threshold_high: float = 0.98,
        min_edge_tokens: int = 64,
        tile_size: int = 512,
        tile_overlap: int = 128,
        # New config-based API (takes precedence when provided)
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        if optimization_config is None:
            # Build config from individual parameters
            optimization_config = OptimizationConfig(
                flash_attention=True,
                tiled_refiner=True,
                disable_cudnn_benchmark=True,
                cache_clearing=True,
                token_routing=use_token_routing,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                edge_threshold_low=edge_threshold_low,
                edge_threshold_high=edge_threshold_high,
                min_edge_tokens=min_edge_tokens,
            )

        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            use_refiner=use_refiner,
            optimization_config=optimization_config,
        )

    def _create_model(self) -> OptimizedGreenFormer:
        # Verify SDPA availability (guaranteed for PyTorch 2.0+)
        assert hasattr(F, "scaled_dot_product_attention"), (
            "PyTorch 2.0+ is required for F.scaled_dot_product_attention (FlashAttention). "
            f"Current version: {__import__('torch').__version__}"
        )

        return OptimizedGreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
            optimization_config=self.config,
        )

    def _report_load_results(self, missing: list[str], unexpected: list[str]) -> None:
        """Handle expected LTRM missing keys gracefully."""
        ltrm_missing = [k for k in missing if "ltrm" in k]
        other_missing = [k for k in missing if "ltrm" not in k]

        if ltrm_missing:
            print(
                f"[Optimized] Expected new LTRM keys not in checkpoint ({len(ltrm_missing)} keys) -- using zero-init."
            )
        if other_missing:
            print(f"[Warning] Missing non-LTRM keys: {other_missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")
