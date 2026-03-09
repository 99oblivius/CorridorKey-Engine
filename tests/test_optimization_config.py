"""Tests for OptimizationConfig and PerformanceMetrics."""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch

from CorridorKeyModule.optimization_config import (
    OptimizationConfig,
    PerformanceMetrics,
    StageMetric,
)

# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


class TestOptimizationConfigProfiles:
    def test_default_equals_original(self):
        assert OptimizationConfig() == OptimizationConfig.original()

    def test_original_all_off(self):
        cfg = OptimizationConfig.original()
        assert cfg.flash_attention is False
        assert cfg.tiled_refiner is False
        assert cfg.disable_cudnn_benchmark is False
        assert cfg.cache_clearing is False
        assert cfg.token_routing is False
        assert cfg.enable_metrics is False

    def test_optimized_profile(self):
        cfg = OptimizationConfig.optimized()
        assert cfg.flash_attention is True
        assert cfg.tiled_refiner is True
        assert cfg.disable_cudnn_benchmark is True
        assert cfg.cache_clearing is True
        assert cfg.token_routing is False

    def test_experimental_profile(self):
        cfg = OptimizationConfig.experimental()
        assert cfg.flash_attention is True
        assert cfg.tiled_refiner is True
        assert cfg.disable_cudnn_benchmark is True
        assert cfg.cache_clearing is True
        assert cfg.token_routing is True

    def test_from_profile_valid_names(self):
        for name in ("original", "optimized", "experimental"):
            cfg = OptimizationConfig.from_profile(name)
            assert isinstance(cfg, OptimizationConfig)

    def test_from_profile_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown optimization profile"):
            OptimizationConfig.from_profile("turbo")


class TestOptimizationConfigImmutability:
    def test_frozen(self):
        cfg = OptimizationConfig.original()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.flash_attention = True  # type: ignore[misc]

    def test_replace_override(self):
        base = OptimizationConfig.original()
        tweaked = dataclasses.replace(base, flash_attention=True, tiled_refiner=True)
        assert tweaked.flash_attention is True
        assert tweaked.tiled_refiner is True
        assert tweaked.cache_clearing is False  # unchanged

    def test_replace_preserves_type(self):
        cfg = dataclasses.replace(OptimizationConfig.optimized(), enable_metrics=True)
        assert isinstance(cfg, OptimizationConfig)


class TestOptimizationConfigDefaults:
    def test_tile_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.tile_size == 512
        assert cfg.tile_overlap == 128

    def test_routing_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.edge_threshold_low == 0.02
        assert cfg.edge_threshold_high == 0.98
        assert cfg.min_edge_tokens == 64


class TestOptimizationConfigHelpers:
    def test_active_optimizations_original(self):
        cfg = OptimizationConfig.original()
        assert cfg.active_optimizations() == []

    def test_active_optimizations_optimized(self):
        cfg = OptimizationConfig.optimized()
        active = cfg.active_optimizations()
        assert len(active) == 4
        assert "flash_attention" in active
        assert any("tiled_refiner" in a for a in active)
        assert "disable_cudnn_benchmark" in active
        assert "cache_clearing" in active

    def test_active_optimizations_single(self):
        cfg = OptimizationConfig(flash_attention=True)
        active = cfg.active_optimizations()
        assert active == ["flash_attention"]

    def test_summary_original(self):
        cfg = OptimizationConfig.original()
        assert "no optimizations" in cfg.summary()

    def test_summary_optimized(self):
        cfg = OptimizationConfig.optimized()
        summary = cfg.summary()
        assert "flash_attention" in summary
        assert "tiled_refiner" in summary


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_measure_records_timing(self):
        metrics = PerformanceMetrics()
        with metrics.measure("test_stage", torch.device("cpu")):
            time.sleep(0.01)
        assert len(metrics.stages) == 1
        assert metrics.stages[0].name == "test_stage"
        assert metrics.stages[0].duration_ms >= 5  # allow some slack

    def test_measure_multiple_stages(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stage_a", torch.device("cpu")):
            pass
        with metrics.measure("stage_b", torch.device("cpu")):
            pass
        assert len(metrics.stages) == 2
        assert metrics.stages[0].name == "stage_a"
        assert metrics.stages[1].name == "stage_b"

    def test_finalize_sums_durations(self):
        metrics = PerformanceMetrics()
        with metrics.measure("a", torch.device("cpu")):
            time.sleep(0.01)
        with metrics.measure("b", torch.device("cpu")):
            time.sleep(0.01)
        metrics.finalize(torch.device("cpu"))
        assert metrics.total_duration_ms >= 10

    def test_summary_format(self):
        metrics = PerformanceMetrics()
        with metrics.measure("encode", torch.device("cpu")):
            pass
        metrics.finalize(torch.device("cpu"))
        summary = metrics.summary()
        assert "encode" in summary
        assert "TOTAL" in summary
        assert "ms" in summary

    def test_cpu_no_vram(self):
        metrics = PerformanceMetrics()
        with metrics.measure("cpu_stage", torch.device("cpu")):
            pass
        assert metrics.stages[0].vram_before_mb == 0.0
        assert metrics.stages[0].vram_after_mb == 0.0
        assert metrics.stages[0].vram_peak_mb == 0.0


class TestStageMetric:
    def test_defaults(self):
        m = StageMetric(name="test")
        assert m.name == "test"
        assert m.duration_ms == 0.0
        assert m.vram_before_mb == 0.0
        assert m.vram_after_mb == 0.0
        assert m.vram_peak_mb == 0.0
