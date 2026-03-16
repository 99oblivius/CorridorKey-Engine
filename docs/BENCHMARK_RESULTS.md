# CorridorKey 4K Benchmark -- Tears of Steel

## Test Configuration

- **Source footage**: Tears of Steel (scene 02_3c) -- CC-BY 3.0 (c) Blender Foundation | mango.blender.org
- **Format**: OpenEXR 16-bit half-float, linear color space
- **Resolution**: 4096x2160 (DCI 4K)
- **Frames**: 100 (24 fps, ~4.2 seconds)
- **Model input size**: 2048x2048
- **GPU**: NVIDIA GeForce RTX 4090 (23.5 GB)
- **Alpha hints**: HSV chroma key (auto-generated from green screen footage)
- **Color pipeline**: `input_is_linear=True` -- engine handles linear-to-sRGB conversion internally

---

## Head-to-Head Comparison

| Metric | Original (fp32, no optimizations) | Optimized (fp16, flash+tiled+cache) | Experimental (fp16, torch.compile reduce-overhead) | Best vs Original |
|---|---:||---:||---:||---:|
| Total time | 285.3 s | 246.1 s | 421.2 s | -13.8% |
| Effective FPS | 0.35 fps | 0.41 fps | 0.24 fps | +17.1% |
| Avg frame time | 1074.2 ms | 696.8 ms | 2258.9 ms | -35.1% |
| Median frame time | 1050.4 ms | 693.9 ms | 368.6 ms | -64.9% |
| Min frame time | 1037.4 ms | 685.6 ms | 364.5 ms | -64.9% |
| Max frame time | 2293.9 ms | 964.4 ms | 121406.8 ms | -58.0% |

### GPU Memory

| Metric | Original (fp32, no optimizations) | Optimized (fp16, flash+tiled+cache) | Experimental (fp16, torch.compile reduce-overhead) | Best vs Original |
|---|---:||---:||---:||---:|
| Dedicated VRAM (physical) | 10415.9 MB | 6653.4 MB | 5159.7 MB | -50.5% |
| PyTorch allocator reserved | 5754.0 MB | 3704.0 MB | 2124.0 MB | -63.1% |
| Idle after model load | 3737.8 MB | 3494.7 MB | 3538.6 MB | -6.5% |

### Warmup Effect (first 5 vs last 5 frames)

| Profile | First 5 avg (ms) | Last 5 avg (ms) | Warmup overhead |
|---|---:|---:|---:|
| Original (fp32, no optimizations) | 1297 | 1048 | +23.7% |
| Optimized (fp16, flash+tiled+cache) | 749 | 692 | +8.2% |
| Experimental (fp16, torch.compile reduce-overhead) | 38155 | 369 | +10234.6% |

---

## Output Files

### Original (fp32, no optimizations)

- **Composite**: `Output/comp_original`
- **Alpha matte**: `Output/alpha_original`

### Optimized (fp16, flash+tiled+cache)

- **Composite**: `Output/comp_optimized`
- **Alpha matte**: `Output/alpha_optimized`

### Experimental (fp16, torch.compile reduce-overhead)

- **Composite**: `Output/comp_experimental`
- **Alpha matte**: `Output/alpha_experimental`

> Compare the composite and alpha EXR sequences side-by-side to evaluate quality
> differences between profiles. Output is linear premultiplied RGBA EXR -- ready
> for compositing in Nuke/Fusion/etc.

---

## Configuration Details

### Original (fp32, no optimizations)

- **Config**: OptimizationConfig: compile_none
- **Active optimizations**: compile_none
- **Model precision**: float32
- **Compile mode**: none
- **Model load time**: 685 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 288.0 s

### Optimized (fp16, flash+tiled+cache)

- **Config**: OptimizationConfig: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing, model_float16, tf32_matmul, compile_none
- **Active optimizations**: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing, model_float16, tf32_matmul, compile_none
- **Model precision**: float16
- **Compile mode**: none
- **Model load time**: 661 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 248.7 s

### Experimental (fp16, torch.compile reduce-overhead)

- **Config**: OptimizationConfig: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing, model_float16, tf32_matmul, token_routing(edge=0.02-0.98)
- **Active optimizations**: flash_attention, tiled_refiner(512x512/128px), disable_cudnn_benchmark, cache_clearing, model_float16, tf32_matmul, token_routing(edge=0.02-0.98)
- **Model precision**: float16
- **Compile mode**: default
- **Model load time**: 1117 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 429.3 s

---

## Analysis

Processing 100 frames of 4K EXR footage (Tears of Steel):

**Optimized (fp16, flash+tiled+cache)** vs Original:
- **Speed**: 0.41 fps vs 0.35 fps (faster by 13.8%)
- **Reserved VRAM**: 3704 MB vs 5754 MB (**-36%**)
- **Total time**: 246.1s vs 285.3s

**Experimental (fp16, torch.compile reduce-overhead)** vs Original:
- **Speed**: 0.24 fps vs 0.35 fps (slower by 47.6%)
- **Reserved VRAM**: 2124 MB vs 5754 MB (**-63%**)
- **Total time**: 421.2s vs 285.3s
