# CorridorKey Engine — Architecture Overview

> For VRAM optimization details see [VRAM_OPTIMIZATIONS.md](VRAM_OPTIMIZATIONS.md).
> For the full async pipeline diagram see [async_pipeline_flowchart.md](async_pipeline_flowchart.md).

---

## 1. Package Layout

```
ck_engine/
  api/            Protocol contract — types, events, errors, frame ranges
  engine/         Engine server — dispatcher, job runner, model pool, event bus
  transport/      stdio + TCP transports with Content-Length framing
  pipeline/       Core processing — generate-alphas and inference pipelines
  generators/     Alpha generator implementations (BiRefNet, GVM, VideoMaMa)
  client.py       Python client library (EngineClient.spawn())
  settings.py     Global and per-project settings (TOML)
backend/          Backwards-compatible shim — re-exports from ck_engine

CorridorKeyModule/
  base_engine.py          _BaseCorridorKeyEngine — abstract base, DMA, cuDNN
  inference_engine.py     CorridorKeyEngine — original torch engine
  optimized_engine.py     OptimizedCorridorKeyEngine — FlashAttention + tiled refiner
  engine_factory.py       create_engine() factory, _MLXEngineAdapter
  optimization_config.py  OptimizationConfig profiles and PerformanceMetrics
  core/
    model_transformer.py  GreenFormer — Hiera backbone + multiscale decoder + CNN refiner
    optimized_model.py    OptimizedGreenFormer, TiledCNNRefiner, LTRM, TokenRouter
    color_utils.py        Despill, sRGB/linear conversion, compositing, despeckle

tui/              Textual TUI — pure engine client, no direct backend imports
tests/            332 tests (protocol, transport, engine, client, CLI, e2e)
```

---

## 2. Engine Process Model

The engine runs as a standalone process reachable over stdio or TCP:

```
Client (TUI / CLI / Plugin)
        |  JSON-RPC 2.0
        v
   Transport layer          stdio or TCP with Content-Length framing
        |
        v
   JSON-RPC Dispatcher      routes methods to handlers (engine.*, job.*, model.*, events.*)
        |
        v
   Job Runner               single active job at a time; queues cancel requests
        |
        v
   Pipeline                 generate-alphas  or  inference
        |
        v
   Model Pool               keeps models resident in VRAM between jobs;
                            reuses a loaded model when the config hash is unchanged,
                            otherwise unloads, resets torch.compile/dynamo state, reloads
        |
        v
   Event Bus                pushes structured JSON-RPC notifications back to subscribers
                            (event.job.*, event.model.*, event.log)
```

Only one job runs at a time. A `job.cancel` request sets a cancellation flag that the pipeline checks between frames; the engine returns to idle and emits `event.job.cancelled`.

---

## 3. Model Hierarchy

### Engine classes

```
_BaseCorridorKeyEngine          (base_engine.py)
    Constructor, checkpoint loading, process_frame() orchestration,
    cuDNN disable, pinned DMA buffer management, PerformanceMetrics
    |
    |-- CorridorKeyEngine       (inference_engine.py)
    |       Wraps GreenFormer directly.
    |       Defaults to OptimizationConfig.original() (all opts off).
    |
    |-- OptimizedCorridorKeyEngine  (optimized_engine.py)
            Wraps OptimizedGreenFormer.
            Defaults to OptimizationConfig.optimized() (FA + tiled refiner + cache clearing).
            Adds LTRM weight handling for token routing.
```

Optimizations are config-driven, not engine-driven. Both engines accept any `OptimizationConfig`; the base `GreenFormer` model applies FlashAttention patching, tiled refiner, and CUDA cache clearing according to whichever config it receives.

### Factory and adapter

`create_engine()` in `engine_factory.py` is the single entry point for all callers:

- Resolves backend: CLI flag > `CORRIDORKEY_BACKEND` env var > auto-detect
- Auto-detect: Apple Silicon + `corridorkey_mlx` installed + `.safetensors` found → `mlx`; any CUDA GPU → `torch_optimized`; otherwise → `torch`
- Returns `OptimizedCorridorKeyEngine`, `CorridorKeyEngine`, or `_MLXEngineAdapter`

`_MLXEngineAdapter` wraps `CorridorKeyMLXEngine` (the optional Apple Silicon package) to match the Torch output contract: converts uint8 MLX outputs to float32, and applies despill and despeckle in the adapter layer since the MLX package stubs those operations.

---

## 4. Alpha Generators

Alpha generators produce coarse matte hints used as input to the keying model. They implement the `AlphaGenerator` protocol defined in `alpha_generators/base.py`:

```python
class AlphaGenerator(Protocol):
    def generate(self, frames: list[Path], output_dir: Path, **kwargs) -> None: ...
```

Three implementations ship with the engine:

| Generator | Module | Mode | VRAM |
|-----------|--------|------|------|
| BiRefNet | `ck_engine/generators/birefnet/` | Per-frame (salient object) | ~4 GB |
| GVM | `ck_engine/generators/gvm/` | Temporal (video-aware) | ~80 GB |
| VideoMaMa | `ck_engine/generators/videomama/` | Temporal (diffusion-based) | ~80 GB |

BiRefNet processes frames independently and is downloaded automatically via torchhub. GVM and VideoMaMa are temporally consistent but require high-end multi-GPU hardware; weights must be downloaded separately. The `generate-alphas` subcommand and `job.submit` with `type=generate` both route through this layer.

---

## 5. Async Inference Pipeline

The inference pipeline overlaps four stages across threads so that disk I/O, GPU inference, DMA transfer, and file writing proceed concurrently:

| Stage | Executor | Work |
|-------|----------|------|
| 1. Read | ThreadPoolExecutor (`cpu_count // 4`) | `cv2.imread` → float32, raw resolution |
| 2. Infer | 1 thread per GPU | GPU upload, resize, normalize, forward pass, GPU post-process |
| 3. DMA drain | 1 thread per GPU | CUDA event sync, pinned-buffer copy → numpy |
| 4. Write | ThreadPoolExecutor (`cpu_count // 4`) | `cv2.imwrite` EXR/PNG per output channel |

Work queues (`work_q`, `write_q`) connect the stages. `work_q` is bounded (`num_gpus * 8`) to throttle readers when GPUs fall behind; a RAM-aware reader pause prevents system memory exhaustion on large clip sets. DMA uses 2-3 pinned CPU buffers (configurable via `--dma-buffers`) so GPU→CPU transfers overlap with the next frame's forward pass.

For the complete flowchart, concurrency diagram, GIL analysis, and backpressure details see [async_pipeline_flowchart.md](async_pipeline_flowchart.md).

---

## 6. Frontend Architecture

The TUI (`tui/`) is a pure engine client built with Textual. It has no direct imports from `CorridorKeyModule` or `ck_engine.pipeline`.

`tui/client.py` contains `TUIEngineClient`, which:

1. Calls `EngineClient.spawn()` to start the engine as a subprocess over stdio
2. Runs a background thread that reads JSON-RPC event notifications from the engine
3. Converts each event to a Textual `Message` subclass (`EngineProgress`, `EngineClipStarted`, `EngineJobCompleted`, etc.) and posts it to the Textual event loop

TUI screens and widgets react to these messages through Textual's normal `on_*` handler mechanism. No pipeline code runs in the TUI process.

The CLI (`ck_engine/`) uses the same `EngineClient` path, making the TUI and CLI interchangeable from the engine's perspective.

---

## 7. Dependency Direction

```
tui/                  imports only  ck_engine.client  (via JSON-RPC)
                              |
ck_engine/            imports       CorridorKeyModule/
                              |
CorridorKeyModule/    imports       torch, numpy, cv2 (no ck_engine imports)
```

Dependencies flow strictly downward. `CorridorKeyModule` has no knowledge of `ck_engine`, `tui`, or the JSON-RPC protocol. `tui` has no knowledge of the model or pipeline internals. This ensures the model layer can be used standalone (e.g. from `engine_factory.create_engine()` directly) without pulling in the server stack, and the TUI can be replaced or extended without touching inference code.
