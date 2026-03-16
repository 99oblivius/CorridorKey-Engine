# CorridorKey Engine

Neural network green screen keying for professional VFX pipelines.
Fork of [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey) with async multi-GPU inference, optimization profiles, a JSON-RPC engine API, and a Textual TUI.

---

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/99oblivius/CorridorKey-Engine.git && cd CorridorKey-Engine
uv sync
```

Windows: run `tools/Install_CorridorKey_Windows.bat` instead.

## Models

```bash
# CorridorKey (required, ~300 MB)
uv run hf download nikopueringer/CorridorKey_v1.0 --local-dir CorridorKeyModule/checkpoints

# BiRefNet — downloaded automatically via torchhub

# GVM (optional, ~80 GB VRAM)
uv run hf download geyongtao/gvm --local-dir ck_engine/generators/gvm/weights

# VideoMaMa (optional, ~80 GB VRAM)
uv run hf download SammyLim/VideoMaMa --local-dir ck_engine/generators/videomama/checkpoints/VideoMaMa
uv run hf download stabilityai/stable-video-diffusion-img2vid-xt \
  --local-dir ck_engine/generators/videomama/checkpoints/stable-video-diffusion-img2vid-xt \
  --include "feature_extractor/*" "image_encoder/*" "vae/*" "model_index.json"
```

## Quick Start

```bash
# TUI (default)
corridorkey-engine
corridorkey-engine /path/to/clips

# Headless CLI
corridorkey-engine inference /path/to/clips --srgb --despill 5 --despeckle --refiner 1.0
corridorkey-engine generate-alphas /path/to/clips --model birefnet

# Engine server (for integrations)
corridorkey-engine serve                     # stdio
corridorkey-engine serve --listen :9400      # TCP daemon
```

### Engine API

CorridorKey runs as a standalone process speaking JSON-RPC 2.0. Any language
can connect — spawn as a subprocess (stdio) or connect to a daemon (TCP).

```python
from ck_engine.client import EngineClient
from ck_engine.api.types import InferenceParams, InferenceSettings

with EngineClient.spawn() as engine:
    job_id = engine.submit_inference(InferenceParams(
        path="/path/to/clips",
        settings=InferenceSettings(despill_strength=0.5),
    ))
    for event in engine.iter_events():
        print(event)
        if type(event).__name__ in ("JobCompleted", "JobFailed"):
            break
```

See [Engine Protocol Reference](docs/engine_protocol.md) for the full spec,
and [examples/](docs/examples/) for complete stdio and TCP client scripts.

## Outputs

| Folder | Format | Contents |
|---|---|---|
| `Matte/` | EXR | Linear alpha |
| `FG/` | EXR | Straight foreground (sRGB gamut) |
| `Processed/` | EXR | Premultiplied linear RGBA |
| `Comp/` | EXR/PNG | Composite preview (transparent RGBA or checkerboard) |

## VRAM at a Glance

| Profile | Precision | VRAM | Warmup | Key features |
|---|---|---|---|---|
| `optimized` (default) | fp16 | ~2-3 GB | ~10-15s | Flash attention, tiled refiner, cache clearing |
| `original` | fp32 | ~9-10 GB | ~5s | No tiling, no cache clearing |
| `performance` | fp16 | ~8-12 GB | ~5-10 min | Full refiner, cuDNN benchmark, max-autotune |

Warmup is first-frame compilation time. Cached after the first run (`~/.cache/torch/inductor/`).

| Add-on | VRAM |
|---|---|
| + GPU postprocessing | +~1.5 GB |
| + cuDNN auto-tune | +2-5 GB |
| BiRefNet alpha hints | ~4 GB |
| GVM / VideoMaMa alpha hints | ~80 GB |

8 GB GPU sufficient for default profile. See [VRAM & Optimization Guide](docs/VRAM_OPTIMIZATIONS.md) for benchmarks and tuning.

## Documentation

| Doc | Audience |
|---|---|
| [CLI Reference](docs/cli_reference.md) | All flags, commands, profiles, multi-GPU, MLX |
| [Engine Protocol](docs/engine_protocol.md) | JSON-RPC spec for plugin/integration developers |
| [Architecture](docs/architecture.md) | Package structure, model hierarchy, pipeline design |
| [VRAM & Optimization](docs/VRAM_OPTIMIZATIONS.md) | Benchmarks, optimization profiles, VRAM breakdown |
| [Async Pipeline](docs/async_pipeline_flowchart.md) | Threading model, DMA pipeline, GIL analysis |
| [Python Examples](docs/examples/) | Complete stdio and TCP client scripts |

## Tests

```bash
uv sync --group dev
uv run pytest                # all tests (no GPU or weights needed)
uv run pytest -m "not gpu"   # skip CUDA tests
```

## License

[CC-BY-NC-SA-4.0 with additional terms](LICENSE) by Corridor Digital. Commercial use of the tool is permitted. Repackaging, paid APIs, or integration into commercial software requires agreement. Forks must retain the "Corridor Key" name.

## Acknowledgements

- [Corridor Digital](https://github.com/nikopueringer/CorridorKey) -- original model and codebase
- [GVM](https://github.com/aim-uofa/GVM) (AIM, Zhejiang University) -- BSD-2-Clause
- [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) (CVLAB, KAIST) -- CC BY-NC 4.0
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) -- MIT

[Corridor Creates Discord](https://discord.gg/zvwUrdWXJm)
