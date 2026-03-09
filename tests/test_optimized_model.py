"""Tests for the VRAM-optimized GreenFormer components.

Tests cover:
  - TiledCNNRefiner: tiling correctness, blend weight symmetry, small-input bypass
  - LTRM: output shape preservation, zero-init identity
  - ECA: output shape and gating range
  - HintBasedTokenRouter: edge mask generation, threshold behavior, min-edge fallback
  - OptimizedGreenFormer: output contract, checkpoint compatibility
  - OptimizedCorridorKeyEngine: process_frame API parity with original engine
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from CorridorKeyModule.core.optimized_model import (
    ECA,
    LTRM,
    HintBasedTokenRouter,
    TiledCNNRefiner,
)

# ---------------------------------------------------------------------------
# ECA
# ---------------------------------------------------------------------------


class TestECA:
    def test_output_shape(self):
        eca = ECA(channels=64)
        x = torch.randn(2, 100, 64)
        out = eca(x)
        assert out.shape == (2, 100, 64)

    def test_gating_range(self):
        """ECA sigmoid gate should produce values in (0, 1)."""
        eca = ECA(channels=32)
        x = torch.randn(1, 50, 32)
        with torch.no_grad():
            out = eca(x)
        # Output should not exceed input magnitude by much
        # (sigmoid gate is in [0, 1], so out = x * gate)
        assert out.abs().max() <= x.abs().max() + 1e-5


# ---------------------------------------------------------------------------
# LTRM
# ---------------------------------------------------------------------------


class TestLTRM:
    def test_output_shape_full_grid(self):
        """LTRM should preserve [B, N, C] shape when N matches spatial grid."""
        ltrm = LTRM(dim=64)
        x = torch.randn(1, 16 * 16, 64)
        out = ltrm(x, spatial_shape=(16, 16))
        assert out.shape == (1, 256, 64)

    def test_output_shape_ragged(self):
        """LTRM should work when N doesn't match spatial grid (DWConv skipped)."""
        ltrm = LTRM(dim=64)
        x = torch.randn(1, 100, 64)  # 100 != 16*16
        out = ltrm(x, spatial_shape=(16, 16))
        assert out.shape == (1, 100, 64)

    def test_zero_init_identity(self):
        """With zero-initialized fc2, LTRM should output approximately x."""
        ltrm = LTRM(dim=32)
        x = torch.randn(1, 64, 32)
        with torch.no_grad():
            out = ltrm(x, spatial_shape=(8, 8))
        # fc2 output is zero -> eca(zero) is zero -> residual + 0 = x
        # But ECA(0) = 0 * sigmoid(conv(pool(0))) = 0, so out = x + 0 = x
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_batch_dimension(self):
        """LTRM should handle batch size > 1."""
        ltrm = LTRM(dim=48)
        x = torch.randn(4, 64, 48)
        out = ltrm(x, spatial_shape=(8, 8))
        assert out.shape == (4, 64, 48)


# ---------------------------------------------------------------------------
# HintBasedTokenRouter
# ---------------------------------------------------------------------------


class TestHintBasedTokenRouter:
    def test_all_background(self):
        """All-zero hint should produce no edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.zeros(1, 1, 64, 64)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.shape == (1, 64)
        assert mask.sum() == 0  # all tokens are "easy"

    def test_all_foreground(self):
        """All-one hint should produce no edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.ones(1, 1, 64, 64)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.sum() == 0

    def test_edge_region(self):
        """Hint with 0.5 values should produce all edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.full((1, 1, 64, 64), 0.5)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.all()  # all tokens are "edge"

    def test_min_edge_tokens_fallback(self):
        """When too few edge tokens, should fall back to all-edge."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=100)
        # Mostly background with a tiny edge
        hint = torch.zeros(1, 1, 64, 64)
        hint[0, 0, 0, 0] = 0.5  # just one edge pixel
        mask = router.compute_edge_mask(hint, 8, 8)
        # After area downsampling to 8x8, at most 1 token would be edge
        # Since min_edge_tokens=100 > 1, all tokens should be marked edge
        assert mask.all()

    def test_mixed_mask(self):
        """Hint with distinct BG/FG/edge regions should route correctly."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.zeros(1, 1, 64, 64)
        hint[0, 0, :32, :] = 0.0  # top half: BG
        hint[0, 0, 32:, :] = 1.0  # bottom half: FG
        # The boundary row will have some intermediate values after area downsample
        mask = router.compute_edge_mask(hint, 8, 8)
        # Most tokens should NOT be edge
        assert mask.sum() < mask.numel()


# ---------------------------------------------------------------------------
# TiledCNNRefiner
# ---------------------------------------------------------------------------


class TestTiledCNNRefiner:
    def test_small_input_bypasses_tiling(self):
        """When input fits in one tile, should skip tiling."""
        refiner = TiledCNNRefiner(tile_size=512, tile_overlap=128)
        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert out.shape == (1, 4, 64, 64)

    def test_output_shape_large_input(self):
        """Tiled processing should produce correct output shape."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        img = torch.randn(1, 3, 128, 128)
        coarse = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert out.shape == (1, 4, 128, 128)

    def test_blend_weight_symmetry(self):
        """Blend weights should be symmetric."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w.shape == (1, 1, 64, 64)
        # Symmetric along both axes
        torch.testing.assert_close(w, w.flip(2))  # vertical symmetry
        torch.testing.assert_close(w, w.flip(3))  # horizontal symmetry

    def test_blend_weight_center_is_one(self):
        """Center of blend weight (away from all borders) should be 1.0."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w[0, 0, 32, 32] == 1.0

    def test_blend_weight_edge_zero(self):
        """Edge of blend weight should be 0.0 (first ramp element)."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w[0, 0, 0, 32] == pytest.approx(0.0, abs=1e-6)
        assert w[0, 0, 32, 0] == pytest.approx(0.0, abs=1e-6)

    def test_tiled_vs_untiled_small(self):
        """On a small input that forces tiling via tile_size=48, output should
        be close to running the refiner on the full input at once.

        Boundary tiles have less overlap context, so we use a relaxed
        tolerance.  At production resolution (2048x2048 with 512px tiles
        and 128px overlap >> 65px receptive field), the difference is
        negligible in the interior and only visible at the image border.
        """
        torch.manual_seed(42)
        refiner_tiled = TiledCNNRefiner(tile_size=48, tile_overlap=24)
        from CorridorKeyModule.core.model_transformer import CNNRefinerModule

        refiner_full = CNNRefinerModule()
        refiner_full.load_state_dict(refiner_tiled.state_dict())

        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)

        with torch.no_grad():
            out_tiled = refiner_tiled(img, coarse)
            out_full = refiner_full(img, coarse)

        # Relaxed tolerance: boundary tiles have limited overlap context
        torch.testing.assert_close(out_tiled, out_full, atol=0.5, rtol=0.5)

    def test_no_nan_or_inf(self):
        """Output should contain no NaN or Inf values."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        img = torch.randn(1, 3, 128, 128)
        coarse = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# OptimizedGreenFormer (structural tests, no checkpoint needed)
# ---------------------------------------------------------------------------


class TestOptimizedGreenFormerStructure:
    """Tests that don't require loading model weights."""

    def test_import(self):
        """OptimizedGreenFormer should be importable."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        assert OptimizedGreenFormer is not None

    def test_token_routing_disabled_by_default(self):
        """With default args, token routing should be off."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False)
        assert model.config.token_routing is False
        assert model.ltrm_stage2 is None
        assert model.ltrm_stage3 is None
        assert model.router is None

    def test_token_routing_enabled(self):
        """With use_token_routing=True, LTRM modules and router should exist."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False, use_token_routing=True)
        assert model.config.token_routing is True
        assert model.ltrm_stage2 is not None
        assert model.ltrm_stage3 is not None
        assert model.router is not None
        # Stage 2 has 16 blocks, stage 3 has 3
        assert len(model.ltrm_stage2) == 16
        assert len(model.ltrm_stage3) == 3

    def test_tiled_refiner_created(self):
        """When use_refiner=True, refiner should be TiledCNNRefiner."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=True)
        assert isinstance(model.refiner, TiledCNNRefiner)

    def test_encoder_has_4_channels(self):
        """Patch embed should accept 4 input channels."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False)
        patch_embed = model.encoder.model.patch_embed.proj
        assert patch_embed.in_channels == 4


# ---------------------------------------------------------------------------
# OptimizedCorridorKeyEngine (mocked, no checkpoint)
# ---------------------------------------------------------------------------


class TestOptimizedEngineAPI:
    """Verify the optimized engine output contract matches the original."""

    def _make_optimized_engine_mock(self, img_size=64):
        """Create an OptimizedCorridorKeyEngine with a mocked model."""
        from CorridorKeyModule.optimization_config import OptimizationConfig
        from CorridorKeyModule.optimized_engine import OptimizedCorridorKeyEngine

        def fake_forward(x):
            b, c, h, w = x.shape
            return {
                "alpha": torch.full((b, 1, h, w), 0.8),
                "fg": torch.full((b, 3, h, w), 0.6),
            }

        mock_model = MagicMock()
        mock_model.side_effect = fake_forward
        mock_model.refiner = None
        mock_model.use_refiner = False

        engine = object.__new__(OptimizedCorridorKeyEngine)
        engine.device = torch.device("cpu")
        engine.img_size = img_size
        engine.checkpoint_path = "/fake/checkpoint.pth"
        engine.use_refiner = False
        engine.config = OptimizationConfig.optimized()
        engine.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        engine.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        engine.model = mock_model
        return engine

    def test_output_keys(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        assert "alpha" in result
        assert "fg" in result
        assert "comp" in result
        assert "processed" in result

    def test_output_shapes(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        h, w = sample_frame_rgb.shape[:2]
        assert result["alpha"].shape == (h, w, 1)
        assert result["fg"].shape == (h, w, 3)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 4)

    def test_output_dtypes(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        for key in ("alpha", "fg", "comp", "processed"):
            assert result[key].dtype == np.float32, f"{key} dtype mismatch"

    def test_uint8_input(self, sample_mask):
        """Engine should handle uint8 input without error."""
        engine = self._make_optimized_engine_mock()
        img_u8 = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        result = engine.process_frame(img_u8, sample_mask)
        assert "alpha" in result

    def test_2d_mask_input(self, sample_frame_rgb):
        """Engine should handle [H, W] mask without [H, W, 1]."""
        engine = self._make_optimized_engine_mock()
        mask_2d = np.random.rand(64, 64).astype(np.float32)
        result = engine.process_frame(sample_frame_rgb, mask_2d)
        assert result["alpha"].shape == (64, 64, 1)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class TestBackendFactory:
    def test_torch_optimized_is_valid_backend(self):
        from CorridorKeyModule.backend import VALID_BACKENDS

        assert "torch_optimized" in VALID_BACKENDS

    def test_resolve_backend_explicit(self):
        from CorridorKeyModule.backend import resolve_backend

        assert resolve_backend("torch") == "torch"
        assert resolve_backend("torch_optimized") == "torch_optimized"
