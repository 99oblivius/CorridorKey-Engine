# CorridorKey Engine — CLI Reference

---

## Commands

| Command | Description |
|---|---|
| `corridorkey-engine` | Launch the Textual TUI (default) |
| `corridorkey-engine <path>` | Launch TUI with a project path pre-loaded |
| `corridorkey-engine inference <path>` | Run keying inference on clips with Input + AlphaHint |
| `corridorkey-engine generate-alphas <path> --model <engine>` | Generate alpha hints (`birefnet`, `gvm`, `videomama`) |
| `corridorkey-engine wizard <path>` | Interactive setup wizard |
| `corridorkey-engine serve` | Start engine process (stdio JSON-RPC) |
| `corridorkey-engine serve --listen :9400` | Start engine daemon (TCP JSON-RPC) |

---

## Global Options

All options are global and must be placed before the subcommand.

### Device & Pipeline

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | Compute device: `auto`, `cuda`, `mps`, `cpu` |
| `--backend` | `auto` | Inference backend: `auto`, `torch`, `torch_optimized`, `mlx` |
| `--devices` | — | Comma-separated GPU indices for multi-GPU, e.g. `0,1` |
| `--img-size` | `2048` | Model input resolution (1024 for FullHD, 2048 for 4K) |
| `--read-workers` | `0` | Reader thread pool size (0 = auto) |
| `--write-workers` | `0` | Writer thread pool size (0 = auto) |
| `--precision` | `fp16` | Floating-point precision: `fp16`, `bf16`, `fp32` |
| `--dma-buffers` | `2` | Pinned DMA buffer count (2–3). Approximately 87 MB page-locked RAM each at 1080p |

### Optimization

| Flag | Default | Description |
|---|---|---|
| `--profile` | — | Preset: `optimized`, `original`, `experimental`\*, `performance`\* |
| `--flash-attention` | profile | FlashAttention patching |
| `--tiled-refiner` | profile | Tiled CNN refiner |
| `--cache-clearing` | profile | CUDA cache clearing between frames |
| `--cudnn-benchmark` | off | cuDNN kernel auto-tune. Faster convolutions, +2–5 GB VRAM |
| `--gpu-postprocess` | profile | GPU postprocessing. Faster, +~1.5 GB VRAM |
| `--cpu-postprocess` | — | Force CPU postprocessing |
| `--token-routing` | off | Experimental sparse attention. Improves speed at 4K+ |
| `--compile-mode` | — | `default`, `reduce-overhead`\*, `max-autotune`\*. Longer first-frame warmup |
| `--tile-size` | `512` | Tile size in pixels for the tiled refiner |
| `--tile-overlap` | `128` | Tile overlap in pixels |

\* Experimental — may increase warmup time.

Individual flags override profile defaults.

### Output

| Flag | Default | Description |
|---|---|---|
| `--comp` | `exr` | Composite output format: `exr`, `png`, `none` |
| `--checkerboard` | off | Render an opaque checkerboard comp instead of transparent RGBA |

---

## Inference-Specific Options

These flags apply to the `inference` subcommand.

| Flag | Default | Description |
|---|---|---|
| `--linear` / `--srgb` | prompt | Input colorspace |
| `--despill 0–10` | prompt | Green spill suppression strength |
| `--despeckle` / `--no-despeckle` | prompt | Morphological despeckle (tracking marker removal) |
| `--despeckle-size` | `400` | Minimum pixel area considered for despeckle |
| `--refiner` | prompt | Refiner strength multiplier |

Omitted flags trigger interactive prompts.

---

## `serve` Options

| Flag | Default | Description |
|---|---|---|
| `--listen <addr>` | — | TCP address to listen on, e.g. `:9400` or `0.0.0.0:9400`. Omit to use stdio. |
| `--log-level <level>` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Optimization Profiles

| Profile | Precision | VRAM | Warmup | Description |
|---|---|---|---|---|
| `original` | fp32 | ~9-10 GB | ~5s | No tiling, no cache clearing, full-resolution refiner |
| `optimized` | fp16 | ~2-3 GB | ~10-15s | FlashAttention + tiled refiner + cache clearing (default on CUDA) |
| `experimental` | fp16 | ~2-3 GB | ~60-90s | Adds `torch.compile` with `reduce-overhead` + token routing |
| `performance` | fp16 | ~8-12 GB | ~5-10 min | Full refiner, cuDNN benchmark, `max-autotune` — highest throughput |

Warmup is the first-frame compilation time. The FX graph cache (`~/.cache/torch/inductor/`)
stores compiled kernels, so subsequent runs with the same profile skip most of the warmup.
Expect the full cost on the first run after a PyTorch or profile change.

---

## Output Folder Structure

| Folder | Format | Contents |
|---|---|---|
| `Matte/` | EXR | Linear alpha matte |
| `FG/` | EXR | Straight foreground (sRGB gamut) |
| `Processed/` | EXR | Premultiplied linear RGBA |
| `Comp/` | EXR / PNG | Composite preview (transparent RGBA or checkerboard) |

---

## Device Resolution Order

```
--device flag  >  CORRIDORKEY_DEVICE env var  >  auto (CUDA > MPS > CPU)

--backend flag  >  CORRIDORKEY_BACKEND env var  >  auto (MLX on Apple Silicon, Torch elsewhere)
```

---

## Multi-GPU

```bash
corridorkey-engine inference /path/to/clips --devices 0,1
```

Each GPU loads its own independent model copy. Per-GPU VRAM usage is unchanged; throughput scales linearly with the number of GPUs.

---

## MLX (Apple Silicon)

Install the MLX backend:

```bash
uv pip install corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git
```

Place weights at `CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors`.

MPS support is experimental. Set the following environment variable if you encounter errors from unsupported ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
