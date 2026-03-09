# CorridorKey VRAM Optimizations

Enable CorridorKey 4K inference (4096x2160) on consumer GPUs with 8 GB VRAM.

The original CorridorKey engine OOMs at its native 2048x2048 inference resolution on 8 GB GPUs. Even with Flash Attention enabled (the minimum to avoid the OOM), PyTorch's allocator reserves 9.8 GB and spills into system RAM. This optimization suite reduces that reserved memory to 1.6 GB (84% reduction) while also being 40% faster per frame, enabling full 4K DCI processing on an 8 GB laptop GPU.

---

## Table of Contents

- [Benchmark Results](#benchmark-results)
- [Optimizations Implemented](#optimizations-implemented)
  - [1. Flash Attention Patching](#1-flash-attention-patching)
  - [2. Tiled CNN Refiner](#2-tiled-cnn-refiner)
  - [3. cuDNN Benchmark Disable](#3-cudnn-benchmark-disable)
  - [4. CUDA Cache Clearing](#4-cuda-cache-clearing)
  - [5. Token Routing (Experimental)](#5-token-routing-experimental)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Benchmark Methodology](#benchmark-methodology)
- [Output Files](#output-files)
- [How to Run](#how-to-run)

---

## Benchmark Results

Tested on real 4K green screen footage from [Tears of Steel](https://mango.blender.org/) (CC-BY 3.0, Blender Foundation).

Test setup: 100 frames, 4096x2160 OpenEXR 16-bit half-float, linear color space, NVIDIA RTX 4060 Laptop GPU (8 GB).

### Speed

| Metric | Flash Attention Only (Baseline) | All Optimizations | Improvement |
|---|---:|---:|---:|
| Total time (100 frames) | 1118.8 s | 785.9 s | -29.8% |
| Effective FPS | 0.09 fps | 0.13 fps | +44.4% |
| Avg frame time | 8423 ms | 5038 ms | -40.2% |
| Median frame time | 8402 ms | 4979 ms | -40.7% |

### VRAM

| Metric | Baseline | Optimized | Improvement |
|---|---:|---:|---:|
| Dedicated VRAM (physical) | 8188 MB (maxed out) | 5166 MB | -37% |
| PyTorch allocator reserved | 9792 MB | 1582 MB | -84% |
| Shared GPU memory spillover | ~1604 MB | 0 MB | Eliminated |
| Headroom within 8 GB VRAM | 0 MB | 6606 MB | -- |

> The baseline completely saturates the 8 GB GPU and spills ~1.6 GB into shared GPU memory (system RAM accessed over PCIe), which is significantly slower than dedicated VRAM. The optimized config fits comfortably within dedicated VRAM with over 6 GB of headroom.

### Why "Flash Attention Only" as baseline?

The original engine (no optimizations at all) OOMs immediately at 4K on 8 GB GPUs. Flash Attention is the minimum required optimization to avoid the out-of-memory crash. The baseline uses Flash Attention only to serve as the closest proxy to original behavior while remaining runnable.

---

## Optimizations Implemented

### 1. Flash Attention Patching

Config flag: `flash_attention: True`
Impact: Required to avoid OOM at 4K on 8 GB GPUs. Without this patch, the global attention blocks materialize a full N x N attention matrix, which does not fit in memory.

#### Problem

CorridorKey uses Meta's [Hiera](https://github.com/facebookresearch/hiera) vision transformer as its backbone. Hiera organizes tokens into "mask units" for windowed attention (Stages 0-1), then switches to global attention (Stages 2-3) by setting `num_windows = 1`.

The problem is in how Hiera constructs its Q/K/V tensors. For global attention, it creates tensors with shape `[B, heads, 1, N, head_dim]`, a 5D tensor where the `num_windows` dimension is 1. When this 5D non-contiguous tensor is passed to `F.scaled_dot_product_attention`, PyTorch's SDPA dispatcher silently falls back to the math backend, which materializes the full `N x N` attention matrix in memory instead of using the memory-efficient FlashAttention kernel.

At 2048x2048 input with Hiera's tokenization, `N` is large enough that this math-backend fallback consumes too much VRAM for consumer GPUs.

#### Solution

Monkey-patch the `forward()` method of Hiera's `MaskUnitAttention` on global-attention blocks (where `use_mask_unit_attn == False`). The patch:

1. Squeezes the `num_windows` dimension from Q/K/V tensors
2. Makes them contiguous 4D tensors: `[B, heads, N, head_dim]`
3. Passes them to `F.scaled_dot_product_attention`, which now correctly dispatches to FlashAttention/memory-efficient kernels

Windowed attention blocks (Stages 0-1) are left unmodified since they are already efficient.

Implementation: `CorridorKeyModule/core/optimized_model.py:40-93` (`_patch_hiera_global_attention()`)
Applied in: `CorridorKeyModule/core/model_transformer.py:225-230` (during `GreenFormer.__init__()`)

---

### 2. Tiled CNN Refiner

Config flag: `tiled_refiner: True`, `tile_size: 512`, `tile_overlap: 128`
Impact: Reduces VRAM usage during the refiner stage by processing the input in small tiles instead of the full 2048x2048 resolution at once.

#### Problem

The CNN Refiner (`CNNRefinerModule`) takes a 7-channel input (RGB + coarse alpha + coarse FG predictions) at the full 2048x2048 resolution and runs dilated residual convolution blocks to produce additive "delta logits" that sharpen edges. Processing the entire 2048x2048 input at once consumes significant VRAM in the intermediate feature maps.

#### Solution

Replace the standard refiner with `TiledCNNRefiner`, which processes the input in overlapping tiles:

- Tile size: 512x512 (default)
- Overlap: 128px (default)
- Stride: `tile_size - overlap` = 384px

Each tile is processed independently through the same CNN pipeline. Tile outputs are merged using linear blend weights (ramps from 0 to 1 over the overlap region) to produce seamless results.

This is mathematically lossless because the refiner's receptive field is ~65px (from dilated residual blocks with dilations 1, 2, 4, 8), and the 128px overlap fully covers it. Any pixel's prediction depends only on inputs within 65px, which the overlap guarantees are identical whether processed as part of the full image or a tile.

If the input fits in a single tile (smaller than `tile_size`), tiling overhead is skipped entirely.

Implementation: `CorridorKeyModule/core/optimized_model.py:262-364` (`TiledCNNRefiner`)
Instantiated in: `CorridorKeyModule/core/model_transformer.py:262-275`

---

### 3. cuDNN Benchmark Disable

Config flag: `disable_cudnn_benchmark: True`
Impact: Reduces VRAM used by cuDNN workspace allocations during convolution benchmarking.

#### Problem

When `torch.backends.cudnn.benchmark = True` (PyTorch's default in many setups), cuDNN runs multiple convolution algorithms on the first call to find the fastest one. Each algorithm trial requires allocating workspace memory, which adds to VRAM usage. On memory-constrained GPUs, this benchmark overhead can push memory usage over the limit.

#### Solution

Set `torch.backends.cudnn.benchmark = False`. cuDNN will use its default heuristic-selected algorithm instead of benchmarking. The selected algorithm may be slightly slower for specific convolution shapes, but avoids the workspace memory overhead.

Implementation: `CorridorKeyModule/base_engine.py:52-54`

---

### 4. CUDA Cache Clearing

Config flag: `cache_clearing: True`
Impact: Prevents memory accumulation between pipeline stages.

#### Problem

PyTorch's CUDA caching allocator retains freed GPU memory blocks for potential reuse. While this avoids the overhead of repeated `cudaMalloc`/`cudaFree` calls, it means memory from one pipeline stage remains "reserved" (from the OS perspective) even after the tensors are freed. When the next stage has a different allocation pattern, it allocates additional memory on top of the cached blocks, inflating total reserved memory.

With the encoder, decoder, and refiner stages each having different tensor shapes and sizes, the caching allocator can accumulate reserved memory across all stages simultaneously.

#### Solution

Call `torch.cuda.empty_cache()` at two strategic points in the inference pipeline:

1. Between encoder and decoder (`model_transformer.py:349-351`)
2. Between decoder and refiner (`model_transformer.py:314-315`)

This releases intermediate CUDA allocations back to the OS between stages, so each stage only needs to hold its own tensors rather than the accumulated cache from all previous stages.

---

### 5. Token Routing (Experimental)

Config flag: `token_routing: True`
Status: Experimental, disabled by default. Requires fine-tuning for production use.

#### Concept

Route "easy" tokens (solid foreground/background, as determined by the alpha hint mask) to a lightweight LTRM (Lightweight Token Refinement Module) instead of full global self-attention. Only "edge" tokens (uncertain alpha values between configurable thresholds) go through the expensive O(N^2) global attention.

- Edge tokens: Alpha hint between 0.02 and 0.98 (configurable) -> full attention
- Easy tokens: Alpha hint below 0.02 or above 0.98 -> LTRM at O(N) cost

The LTRM architecture: `LayerNorm -> Linear expand -> GELU -> DWConv 5x5 -> GELU -> Linear project -> ECA residual gating`

The LTRM weights are zero-initialized (`fc2` weights = 0), so the module starts as an identity function. This makes it fully compatible with the pretrained checkpoint without any fine-tuning. The model can be loaded and run with token routing enabled, but optimal quality requires fine-tuning the LTRM weights.

Implementation: `CorridorKeyModule/core/optimized_model.py:101-254` (LTRM, ECA, HintBasedTokenRouter)

---

## Architecture

### Engine Hierarchy

```
_BaseCorridorKeyEngine (base_engine.py)
    Abstract base class: constructor, checkpoint loading,
    process_frame() pipeline, cuDNN disable, metrics
    |
    |--- CorridorKeyEngine (inference_engine.py)
    |       Original engine. Uses GreenFormer directly.
    |       Defaults to OptimizationConfig.original() (all opts off)
    |
    |--- OptimizedCorridorKeyEngine (optimized_engine.py)
            Optimized engine. Uses OptimizedGreenFormer.
            Defaults to OptimizationConfig.optimized() (4 production opts on)
```

### Model Hierarchy

```
GreenFormer (model_transformer.py)
    Base model: Hiera backbone, multiscale decoders, CNN refiner
    Handles: FlashAttention patching, tiled refiner, cache clearing
    |
    |--- OptimizedGreenFormer (optimized_model.py)
            Extends GreenFormer with token routing machinery
            (LTRM + HintBasedTokenRouter)
            When routing is disabled, delegates entirely to GreenFormer.forward()
```

### Design Principle

Optimizations are config-driven, not engine-driven. Both engines accept any `OptimizationConfig`. The `GreenFormer` base model handles FlashAttention, tiled refiner, and cache clearing based on the config, so even the "original" `CorridorKeyEngine` can use these optimizations if given the right config. The `OptimizedCorridorKeyEngine` simply defaults to the optimized profile and adds LTRM weight handling.

### Inference Pipeline

```
Input (4096x2160 EXR, linear float)
  |
  v
[Lanczos4 resize to 2048x2048]
  |
  v
[Linear -> sRGB conversion] (if input_is_linear=True)
  |
  v
[ImageNet normalization + alpha hint concat -> 4-channel input]
  |
  v
[Hiera Encoder]    Stages 0-1: Windowed attention (efficient)
  |                Stages 2-3: Global attention (FlashAttention patched)
  |
  |-- torch.cuda.empty_cache() (if cache_clearing)
  |
  v
[Multiscale Decoder]    Predicts coarse alpha (1ch) + coarse FG (3ch)
  |
  |-- torch.cuda.empty_cache() (if cache_clearing)
  |
  v
[CNN Refiner / TiledCNNRefiner]    7ch input (RGB + coarse predictions)
  |                                Produces additive delta logits
  |                                (512x512 tiles if tiled_refiner)
  v
[Sigmoid activation]
  |
  v
[Lanczos4 resize back to 4096x2160]
  |
  v
[Post-processing: despill, premultiply, composite]
  |
  v
Output: alpha, FG (sRGB), processed (linear premul RGBA), comp (sRGB preview)
```

### Auto-Backend Selection

In `CorridorKeyModule/backend.py`, the system auto-detects the optimal engine:

- CUDA GPU with less than 16 GB VRAM: Uses `OptimizedCorridorKeyEngine` with `OptimizationConfig.optimized()`
- CUDA GPU with 16 GB or more VRAM: Uses standard `CorridorKeyEngine` (no optimizations needed)
- Apple Silicon with MLX available: Uses MLX backend

---

## Configuration

### OptimizationConfig Profiles

| Profile | `flash_attention` | `tiled_refiner` | `disable_cudnn_benchmark` | `cache_clearing` | `token_routing` |
|---|:---:|:---:|:---:|:---:|:---:|
| `original` | off | off | off | off | off |
| `optimized` (production) | on | on | on | on | off |
| `experimental` | on | on | on | on | on |

### CLI Flags

```bash
# Use optimized profile (default for GPUs with less than 16GB)
uv run python corridorkey_cli.py --action run_inference --flash-attention --tiled-refiner --disable-cudnn-benchmark --cache-clearing

# Individual toggles
--flash-attention / --no-flash-attention
--tiled-refiner / --no-tiled-refiner
--tile-size N          # default: 512
--tile-overlap N       # default: 128
--disable-cudnn-benchmark / --no-disable-cudnn-benchmark
--cache-clearing / --no-cache-clearing
--token-routing / --no-token-routing
--metrics              # enable per-stage timing/VRAM reporting
```

### Python API

```python
from CorridorKeyModule.optimization_config import OptimizationConfig
from CorridorKeyModule.optimized_engine import OptimizedCorridorKeyEngine

# Production config (4 optimizations)
config = OptimizationConfig.optimized()

# Custom config
config = OptimizationConfig(
    flash_attention=True,
    tiled_refiner=True,
    tile_size=512,
    tile_overlap=128,
    disable_cudnn_benchmark=True,
    cache_clearing=True,
    enable_metrics=True,
)

engine = OptimizedCorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device="cuda",
    img_size=2048,
    use_refiner=True,
    optimization_config=config,
)

# Process a frame (supports linear EXR input)
result = engine.process_frame(image_rgb, alpha_hint, input_is_linear=True)
# result["alpha"]     -> [H, W, 1] float32 alpha matte
# result["fg"]        -> [H, W, 3] float32 sRGB foreground
# result["processed"] -> [H, W, 4] float32 linear premultiplied RGBA (EXR-ready)
# result["comp"]      -> [H, W, 3] float32 sRGB composite preview
# result["metrics"]   -> PerformanceMetrics (if enable_metrics=True)
```

---

## Benchmark Methodology

### Test Footage

Source: [Tears of Steel](https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/02_3c/linear/) (scene 02_3c)
License: CC-BY 3.0 (c) Blender Foundation | mango.blender.org
Format: OpenEXR 16-bit half-float, 4096x2160 (DCI 4K), linear color space
Frames: First 100 frames (~4.2 seconds at 24 fps)
Content: Green screen footage with actors, real production footage rather than synthetic test data

### Alpha Hint Generation

Alpha hints were auto-generated using HSV chroma keying:

1. Read linear EXR frame
2. Convert linear to sRGB (piecewise transfer function)
3. Convert sRGB to HSV
4. Threshold green hue range (35-85), saturation (>40), value (>30)
5. Invert (green = background, non-green = foreground)
6. Erode with 7px elliptical kernel (slight under-prediction, which the model handles better)
7. Gaussian blur with 21px kernel (produces soft/coarse edges the model expects)
8. Save as uint8 PNG

Script: `tears_of_steel_test/generate_alpha_hints.py`

### Benchmark Script

`benchmark_4k_vram.py` runs each configuration in a separate subprocess to ensure clean GPU state:

1. Flash Attention Only (baseline), the minimum viable config
2. All Optimizations: flash + tiled refiner + cuDNN disable + cache clearing

For each config, it measures:
- Per-frame wall-clock time (ms)
- PyTorch allocator peak allocated/reserved memory
- Device-level GPU memory via `torch.cuda.mem_get_info()` (polled at 25ms intervals in a background thread)
- Per-stage timing (inference, postprocessing) via `PerformanceMetrics`

Output is written as EXR frame sequences (linear premultiplied RGBA + single-channel alpha) for quality comparison.

### GPU Memory Measurement

Three levels of GPU memory are tracked:

| Level | API | What it shows |
|---|---|---|
| Live tensors | `torch.cuda.max_memory_allocated()` | PyTorch tensor memory only |
| Allocator reserved | `torch.cuda.max_memory_reserved()` | Total memory held by PyTorch's caching allocator (includes freed-but-cached blocks) |
| Device-level | `torch.cuda.mem_get_info()` | Actual GPU memory usage including CUDA context, cuDNN workspace, and driver overhead. Equivalent to what Task Manager shows. Capped at physical VRAM and cannot detect shared memory spillover. |

When allocator reserved exceeds physical VRAM, Windows spills into shared GPU memory (system RAM accessed over PCIe), which is dramatically slower. The benchmark reports this spillover explicitly.

---

## Output Files

### Benchmark Outputs

```
Output/
  comp_baseline/          # Processed RGBA EXR sequence (baseline)
  comp_optimized/         # Processed RGBA EXR sequence (optimized)
  alpha_baseline/         # Alpha matte EXR sequence (baseline)
  alpha_optimized/        # Alpha matte EXR sequence (optimized)
  raw_greenscreen_h264.mp4    # Original footage (H.264, for viewing)
  comp_baseline_h264.mp4      # Composite preview (baseline, H.264)
  comp_optimized_h264.mp4     # Composite preview (optimized, H.264)
  alpha_baseline_h264.mp4     # Alpha matte preview (baseline, H.264)
  alpha_optimized_h264.mp4    # Alpha matte preview (optimized, H.264)
```

### EXR Output Format

- Processed RGBA: Linear premultiplied RGBA, half-float, PXR24 compression. Ready for compositing in Nuke, Fusion, After Effects, etc.
- Alpha: Single-channel linear float

### Test Data

```
tears_of_steel_test/
  frames/                 # 100 EXR source frames (4096x2160, ~51 MB each)
  alpha_hints/            # 100 PNG alpha hints (4096x2160, auto-generated)
  download_frames.py      # Downloads frames from media.xiph.org
  generate_alpha_hints.py # HSV chroma key alpha hint generator
```

---

## How to Run

### 1. Download test footage

```bash
uv run python tears_of_steel_test/download_frames.py
```

Downloads the first 100 EXR frames (~5.1 GB) from the Tears of Steel open movie project.

### 2. Generate alpha hints

```bash
uv run python tears_of_steel_test/generate_alpha_hints.py
```

Generates coarse alpha hints using HSV chroma keying.

### 3. Run benchmark

```bash
uv run python benchmark_4k_vram.py
```

Processes all 100 frames through both configurations, generates the report at `benchmark_4k_results.md`, and writes output EXR sequences to `Output/`.

---

## Key Implementation Files

| File | Purpose |
|---|---|
| `CorridorKeyModule/optimization_config.py` | `OptimizationConfig` dataclass, profiles, `PerformanceMetrics` |
| `CorridorKeyModule/base_engine.py` | `_BaseCorridorKeyEngine` abstract base class |
| `CorridorKeyModule/optimized_engine.py` | `OptimizedCorridorKeyEngine` with LTRM weight handling |
| `CorridorKeyModule/core/optimized_model.py` | FlashAttention patch, TiledCNNRefiner, LTRM, ECA, TokenRouter |
| `CorridorKeyModule/core/model_transformer.py` | `GreenFormer` model (applies FA, tiling, cache clearing) |
| `CorridorKeyModule/backend.py` | Auto-backend selection based on GPU VRAM |
| `benchmark_4k_vram.py` | 4K benchmark script |
| `benchmark_4k_results.md` | Latest benchmark results |
