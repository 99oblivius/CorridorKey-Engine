# CorridorKeyModule

The core inference engine — GreenFormer architecture (Hiera backbone + CNN refiner)
with torch.compile, CUDA graphs, TensorRT, tiled processing, and async DMA transfers.

This module is consumed by the `ck_engine` package. For usage, see:
- [Architecture Overview](../docs/architecture.md) — model hierarchy and engine factory
- [VRAM & Optimization Guide](../docs/VRAM_OPTIMIZATIONS.md) — benchmarks and profiles
- [Python Examples](../docs/examples/) — direct engine API usage
