# CorridorKey 4K Benchmark — Tears of Steel

## Test Configuration

- **Source footage**: Tears of Steel (scene 02_3c) — CC-BY 3.0 (c) Blender Foundation | mango.blender.org
- **Format**: OpenEXR 16-bit half-float, linear color space
- **Resolution**: 4096x2160 (DCI 4K)
- **Frames**: 100 (24 fps, ~4.2 seconds)
- **Model input size**: 2048x2048
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
- **Alpha hints**: HSV chroma key (auto-generated from green screen footage)
- **Color pipeline**: `input_is_linear=True` — engine handles linear→sRGB conversion internally

> **Note**: The original engine OOMs at 4K on 8 GB GPUs. Flash Attention is the
> minimum required optimization. The "baseline" config uses Flash Attention only
> to serve as the closest proxy to original behavior while remaining runnable.

---

## Head-to-Head Comparison

| Metric | Flash Attention Only (Baseline) | All Optimizations | Delta |
|---|---:|---:|---:|
| Total time | 1118.8 s | 785.9 s | -333.0 s (-29.8%) |
| Effective FPS | 0.09 fps | 0.13 fps | +0.04 fps (+44.4%) |
| Avg frame time | 8423.2 ms | 5037.5 ms | -3385.7 ms (-40.2%) |
| Median frame time | 8401.7 ms | 4979.1 ms | -3422.6 ms (-40.7%) |
| Min frame time | 8081.4 ms | 4585.9 ms | -3495.5 ms (-43.3%) |
| Max frame time | 10437.2 ms | 6369.2 ms | -4068.0 ms (-39.0%) |

#### GPU Memory

| Metric | Flash Attention Only (Baseline) | All Optimizations | Delta |
|---|---:|---:|---:|
| Dedicated VRAM (physical) | 8187.5 MB | 5165.5 MB | -3022.0 MB (-36.9%) |
| PyTorch allocator reserved | 9792.0 MB | 1582.0 MB | -8210.0 MB (-83.8%) |
| Idle after model load | 1893.5 MB | 1893.5 MB | 0.0 MB (0.0%) |

> **Shared GPU memory spillover (system RAM):** The baseline's PyTorch allocator
> reserved **9792 MB** but the GPU only has **8188 MB** of
> dedicated VRAM — meaning **~1604 MB spilled into shared GPU memory**
> (system RAM accessed over PCIe). This is drastically slower than dedicated VRAM
> and is a major contributor to the baseline's poor performance. The device-level
> peak is capped at 8188 MB because `torch.cuda.mem_get_info()` cannot
> see beyond physical VRAM.
>
> The optimized config reserved only **1582 MB** — well within the
> 8188 MB physical VRAM with **6606 MB headroom**.

### Warmup Effect (first 5 vs last 5 frames)

| | First 5 avg (ms) | Last 5 avg (ms) | Warmup overhead |
|---|---:|---:|---:|
| Baseline | 8698 | 8495 | +2.4% |
| Optimized | 5269 | 4926 | +7.0% |

---

## Output Files

### Flash Attention Only (baseline)

- **Composite**: `Output/comp_baseline`
- **Alpha matte**: `Output/alpha_baseline`

### All Optimizations

- **Composite**: `Output/comp_optimized`
- **Alpha matte**: `Output/alpha_optimized`

> Compare the composite and alpha EXR sequences side-by-side to evaluate quality
> differences between baseline (no tiled refiner) and optimized (tiled refiner).
> Output is linear premultiplied RGBA EXR — ready for compositing in Nuke/Fusion/etc.
> The tiled refiner processes the CNN in 512x512 overlapping tiles, which may
> introduce subtle differences at tile boundaries.

---

## Configuration Details

### Flash Attention Only (baseline)

- **Config**: OptimizationConfig: flash_attention
- **Active optimizations**: flash_attention
- **Model load time**: 1358 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 1126.1 s

### All Optimizations

- **Config**: OptimizationConfig: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing
- **Active optimizations**: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing
- **Model load time**: 1314 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 798.7 s

---

## Analysis

Processing 100 frames of 4K EXR footage (Tears of Steel):

- **Speed**: 0.13 fps optimized vs 0.09 fps baseline (faster by 29.8%)
- **Reserved VRAM**: 1582 MB optimized vs 9792 MB baseline (**-84%**)
- **Total time**: 785.9s optimized vs 1118.8s baseline

The optimizations reduce CUDA reserved memory by **84%** while 
also being **29.8% faster** over sustained multi-frame processing.
