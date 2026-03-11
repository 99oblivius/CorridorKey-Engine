# CorridorKey

Neural network green screen keying for professional VFX pipelines.
Fork of [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey) by Corridor Digital.

## Features

- Physically accurate unmixing of straight-color foreground and linear alpha (32-bit EXR)
- Resolution independent inference (2048x2048 backbone, scales to 4K+)
- Async multi-GPU pipeline with deferred DMA transfers and pipelined reader/writer thread pools
- GPU and CPU postprocessing paths (toggle with `--gpu-postprocess` / `--cpu-postprocess`)
- Transparent RGBA or checkerboard composite output
- Morphological despeckle (tracking marker removal)
- Rich terminal wizard with real-time progress and IO throughput
- Alpha hint generators: BiRefNet (~4 GB), GVM (~80 GB), VideoMaMa (~80 GB)

## Hardware

**This software saturates your CPU & GPU at 100% utilization. Ensure adequate cooling.**

| Component | VRAM |
|---|---|
| CorridorKey (optimized profile) | <8 GB |
| CorridorKey (original, unoptimized) | ~22.7 GB |
| + GPU postprocessing | +~1.5 GB |
| + cuDNN auto-tune (`--cudnn-benchmark`) | +2-5 GB |
| BiRefNet alpha hints | ~4 GB |
| GVM / VideoMaMa alpha hints | ~80 GB |

12 GB GPU recommended. The 2048 profile compilation and 4K inference fits in 8-12 GB. 1024 img_size fits in 2-3 GB.
Running on a GPU that is not driving your display avoids OOM from compositor overhead.

**Windows:** NVIDIA drivers must support CUDA 12.8+.

## Installation

Requires [uv](https://docs.astral.sh/uv/) (handles Python, venvs, and packages).

**Windows:**
1. Clone this repository
2. Run `Install_CorridorKey_Windows.bat`
3. *(Optional)* `Install_GVM_Windows.bat` / `Install_VideoMaMa_Windows.bat`

**Linux / Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/nikopueringer/CorridorKey.git && cd CorridorKey
uv sync
```

**Models:**
```bash
# CorridorKey (required, ~300 MB)
uv run hf download nikopueringer/CorridorKey_v1.0 --local-dir CorridorKeyModule/checkpoints

# BiRefNet -- downloaded automatically via torchhub

# GVM (optional)
uv run hf download geyongtao/gvm --local-dir gvm_core/weights

# VideoMaMa (optional)
uv run hf download SammyLim/VideoMaMa --local-dir VideoMaMaInferenceModule/checkpoints/VideoMaMa
uv run hf download stabilityai/stable-video-diffusion-img2vid-xt \
  --local-dir VideoMaMaInferenceModule/checkpoints/stable-video-diffusion-img2vid-xt \
  --include "feature_extractor/*" "image_encoder/*" "vae/*" "model_index.json"
```

## Usage

### Wizard

```bash
uv run corridorkey wizard "/path/to/footage"
# Windows: drag folder onto CorridorKey_DRAG_CLIPS_HERE_local.bat
```

### Example

Recommendations:
- 1024 for FullHD
- 2048 for 4K footage
- --cpu-postprocess mostly only makes sense for under FullHD resolutions with a slower GPU
- --dma-buffers defaults to 2 but for a small cost in additional video memory can perform faster if the GPU is not bottlenecked

```bash
uv run corridorkey wizard "/path/to/footage" --img-size 1024 --cudnn-benchmark --gpu-postprocess --dma-buffers 3 --no-comp-png --precision fp16
```

### Subcommands

```bash
uv run corridorkey list-clips
uv run corridorkey generate-alphas              # GVM
uv run corridorkey generate-alphas-birefnet     # BiRefNet (~4 GB)
uv run corridorkey run-inference                # prompts for settings
```

### Batch (non-interactive)

```bash
uv run corridorkey run-inference \
  --srgb --despill 5 --despeckle --refiner 1.0 \
  --profile optimized --max-frames 100
```

### Multi-GPU

```bash
uv run corridorkey run-inference --devices 0,1
```

Each GPU loads its own model copy. VRAM per GPU is unchanged; throughput scales linearly.

### Outputs

| Folder | Format | Contents |
|---|---|---|
| `Matte/` | EXR | Linear alpha |
| `FG/` | EXR | Straight foreground (sRGB gamut -- convert to linear for compositing) |
| `Processed/` | EXR | Premultiplied linear RGBA (drop into Premiere/Resolve) |
| `Comp/` | PNG | Composite preview (transparent RGBA or checkerboard) |

## Performance Tuning

### Profiles

| Profile | VRAM | Notes |
|---|---|---|
| `original` | ~22.7 GB | No optimizations |
| `optimized` | ~8 GB | FlashAttention + tiled refiner + cache clearing (default on CUDA) |
| `experimental` | ~8 GB | Adds `torch.compile` `reduce-overhead` |
| `performance` | ~8 GB | `max-autotune` compilation, longer warmup |

### Flags

| Flag | Effect |
|---|---|
| `--gpu-postprocess` / `--cpu-postprocess` | GPU: faster, +~1.5 GB VRAM |
| `--cudnn-benchmark` | cuDNN kernel auto-tune. +2-5 GB VRAM, faster convolutions after warmup |
| `--no-comp-png` | Skip composite PNG output (saves write IO) |
| `--checkerboard` | Opaque checkerboard comp instead of transparent RGBA |
| `--compile-mode MODE` | `default`, `reduce-overhead`, `max-autotune`. (WIP) Longer first-frame warmup |
| `--dma-buffers N` | Pinned DMA buffers for GPU->CPU transfer (2-3). ~87 MB page-locked RAM each at 1080p |
| `--token-routing` | Experimental sparse attention. Can improve speed at 4K+ |
| `--precision PREC` | `fp16` (default), `bf16`, `fp32` |

The pipeline prints a postprocessing hint after each run if it detects you would benefit from switching between GPU and CPU mode.

### Device and backend

**Device:** `--device` flag > `CORRIDORKEY_DEVICE` env var > auto (CUDA > MPS > CPU)

**Backend:** `--backend` flag > `CORRIDORKEY_BACKEND` env var > auto (MLX on Apple Silicon, Torch elsewhere)

MLX setup:
```bash
uv pip install corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git
# Place weights at CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
```

Mac MPS is experimental. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops.

## Tests

```bash
uv sync --group dev
uv run pytest
```

No GPU or model weights required.

## License

[CC-BY-NC-SA-4.0 with additional terms](LICENSE) by Corridor Digital. You may use this tool in commercial projects. You may not repackage/sell it, offer it as a paid API, or integrate into commercial software without agreement. Forks must retain the "Corridor Key" name. Contact: contact@corridordigital.com

## Acknowledgements

- [Corridor Digital](https://github.com/nikopueringer/CorridorKey) -- original CorridorKey model and codebase
- [GVM](https://github.com/aim-uofa/GVM) (AIM, Zhejiang University) -- BSD-2-Clause
- [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) (CVLAB, KAIST) -- CC BY-NC 4.0, model checkpoints under [Stability AI Community License](https://stability.ai/license)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) -- MIT

[Corridor Creates Discord](https://discord.gg/zvwUrdWXJm)
