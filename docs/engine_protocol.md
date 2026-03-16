# CorridorKey Engine — Protocol Reference

A complete reference for the `corridorkey-engine` JSON-RPC 2.0 API.
This document is written for integration developers: if you want to build
a Blender plugin, a DaVinci panel, a render farm controller, or any other
client, this is your starting point.

---

**Contents:**
[Architecture](#architecture) |
[Transport](#transport) |
[Python Quick Start](#python-quick-start) |
[Any-Language Quick Start](#any-language-quick-start) |
[Protocol Methods](#protocol-methods) |
[Event Notifications](#event-notifications) |
[Error Codes](#error-codes) |
[Frame Range Syntax](#frame-range-syntax) |
[Model Caching](#model-caching) |
[Subscription Model](#subscription-model)

---

## Architecture

The engine is a long-running process. Every frontend — TUI, CLI, Blender
plugin — is a client. No Python imports cross the boundary; everything
goes through JSON-RPC messages.

```
                    +-----------------------------+
  TUI  --stdio--+   |  corridorkey-engine         |
                |   |                             |
  CLI  --stdio--+-->|  Transport (stdio / TCP)    |
                |   |       |                     |
  Blender--TCP--+   |       v                     |
                    |  JSON-RPC Dispatcher        |
                    |       |                     |
                    |  +----+------+  +---------+ |
                    |  | JobRunner |  | EventBus| |
                    |  |           |  |(pub/sub)| |
                    |  +----+------+  +---------+ |
                    |       |                     |
                    |  +----+------+              |
                    |  | ModelPool |              |
                    |  | (VRAM $)  |              |
                    |  +----+------+              |
                    |       |                     |
                    |  +----+------------------+  |
                    |  | pipeline/             |  |
                    |  |   generate.py         |  |
                    |  |   inference.py        |  |
                    |  | generators/           |  |
                    |  |   birefnet/gvm/...    |  |
                    |  | CorridorKeyModule/    |  |
                    |  +----------------------+   |
                    +-----------------------------+
```

**Key properties:**

- **One active job at a time.** Submit while busy gets an error (-32000), not a queue position.
- **Stateless jobs, stateful cache.** Each job carries all its settings. The model cache is a performance optimization, never a requirement.
- **Events, not polling.** The engine pushes structured notifications; clients do not need to poll for progress.

**Package layout:**

```
ck_engine/
  api/            Protocol contract (types, events, errors, frame ranges)
  engine/         Engine server (dispatcher, job runner, model pool, event bus)
  transport/      stdio + TCP with Content-Length framing
  pipeline/       Core processing (generate, inference)
  generators/     Alpha generators (BiRefNet, GVM, VideoMaMa)
  client.py       Python client library (EngineClient)
  settings.py     Global and project settings (TOML)
```

---

## Transport

### Mode selection

```bash
# stdio — single client, engine is a subprocess
corridorkey-engine

# TCP daemon — multiple clients, persistent process
corridorkey-engine --listen :9400
```

### Content-Length framing

Both stdio and TCP use the same LSP-style framing. Every message is
preceded by a header block:

```
Content-Length: <byte-length>\r\n
\r\n
<UTF-8 JSON body>
```

The body is a single JSON object with no trailing newline. Byte length is
measured after UTF-8 encoding.

**Example — writing a request from any language:**

```
Content-Length: 55\r\n
\r\n
{"jsonrpc":"2.0","method":"engine.capabilities","id":1}
```

**Example — reading a response:**

1. Read bytes until you see `\r\n\r\n`.
2. Parse the `Content-Length` value from the header.
3. Read exactly that many bytes — that is the JSON body.
4. Repeat.

There is no delimiter between messages. The Content-Length header is the
only framing. Partial reads are normal; buffer and retry until you have
the full body.

### stdio notes

- Engine reads from **stdin**, writes to **stdout**.
- **stderr** is reserved for Python logging / diagnostics. Never parse it.
- Send requests to stdin, read responses and events from stdout.
- A single reader thread on stdout handles both responses (matched by `id`)
  and event notifications (no `id` field).

### TCP notes

- Each accepted connection is a single client session (one client per connection).
- Same Content-Length framing as stdio.
- Connect, exchange messages, disconnect. The engine keeps running after a client disconnects.

---

## Python Quick Start

Install and import from `ck_engine`:

```python
from ck_engine.client import EngineClient
from ck_engine.api.types import GenerateParams, InferenceParams, InferenceSettings
```

### Spawn a subprocess client

`EngineClient.spawn()` starts `corridorkey-engine` as a subprocess and
connects via stdio. The `with` block shuts the engine down on exit.

```python
with EngineClient.spawn() as engine:
    # Inspect engine capabilities
    caps = engine.capabilities()
    print(caps.version, caps.devices)

    # Discover clips — no job submission, read-only
    clips = engine.scan_project("/path/to/clips")
    print(f"Found {len(clips)} clips")

    # Generate alpha mattes
    job_id = engine.submit_generate(GenerateParams(
        path="/path/to/clips",
        model="birefnet",
        mode="fill",         # "replace" | "fill" | "skip"
        frames=None,         # None = all frames
    ))
    for event in engine.iter_events():
        print(event)
        if type(event).__name__ in ("JobCompleted", "JobFailed", "JobCancelled"):
            break

    # Run inference
    job_id = engine.submit_inference(InferenceParams(
        path="/path/to/clips",
        settings=InferenceSettings(
            input_is_linear=False,
            despill_strength=0.5,
            auto_despeckle=True,
            despeckle_size=400,
            refiner_scale=1.0,
        ),
    ))
    for event in engine.iter_events():
        print(event)
        if type(event).__name__ in ("JobCompleted", "JobFailed", "JobCancelled"):
            break
```

### Connect to a running daemon

Use `EngineClient.connect()` when the engine is already running as a TCP daemon.

```python
# Engine started separately: corridorkey-engine --listen :9400

with EngineClient.connect("localhost:9400") as engine:
    caps = engine.capabilities()
    job_id = engine.submit_inference(InferenceParams(
        path="/path/to/clips",
        settings=InferenceSettings(despill_strength=0.3),
    ))
    for event in engine.iter_events():
        print(event)
        if type(event).__name__ in ("JobCompleted", "JobFailed"):
            break
```

### Event iteration threading model

`iter_events()` blocks on transport reads. Run it in a background thread
if your application has its own event loop:

```python
import threading

engine = EngineClient.spawn()
engine.subscribe(["job", "model"])

def event_loop():
    for event in engine.iter_events():
        handle(event)

t = threading.Thread(target=event_loop, daemon=True)
t.start()

job_id = engine.submit_inference(params)
# ... do other work ...
engine.shutdown()
engine.close()
```

Internally, a single reader thread dispatches responses to waiting
`Future` objects (matched by `id`) and forwards notifications to the
event queue consumed by `iter_events()`.

---

## Any-Language Quick Start

You only need to speak JSON over a byte pipe.

### Step 1 — Start the engine

```bash
# subprocess mode (stdio)
corridorkey-engine

# daemon mode (TCP)
corridorkey-engine --listen :9400
```

### Step 2 — Send framed JSON-RPC requests

Write to stdin (subprocess) or the TCP socket. Use Content-Length framing.

```
Content-Length: 55\r\n
\r\n
{"jsonrpc":"2.0","method":"engine.capabilities","id":1}
```

```
Content-Length: 57\r\n
\r\n
{"jsonrpc":"2.0","method":"project.scan","id":2,"params":{"path":"/clips"}}
```

```
Content-Length: 93\r\n
\r\n
{"jsonrpc":"2.0","method":"job.submit","id":3,"params":{"type":"generate","path":"/clips","model":"birefnet"}}
```

### Step 3 — Read framed responses and events

Both responses and event notifications arrive on the same stream.
Distinguish them by the presence of the `id` field:

- **Response** — has `id`, has `result` or `error`. Match to your pending request.
- **Notification** — no `id`, has `method` starting with `event.`. These are pushed asynchronously.

```json
// Response to id=1
{"jsonrpc":"2.0","id":1,"result":{"version":"2.0.0","generators":["birefnet","gvm","videomama"],...}}

// Notification (no id)
{"jsonrpc":"2.0","method":"event.job.accepted","params":{"job_id":"j-a1b2c3","type":"generate","total_frames":150}}

{"jsonrpc":"2.0","method":"event.job.progress","params":{"job_id":"j-a1b2c3","clip":"plate_001","done":45,"total":150,"fps":12.3}}

{"jsonrpc":"2.0","method":"event.job.completed","params":{"job_id":"j-a1b2c3","clips_ok":1,"clips_failed":0,"total_frames":150,"frames_ok":150,"frames_failed":0,"elapsed_seconds":12.2}}
```

---

## Protocol Methods

All requests follow JSON-RPC 2.0: `{"jsonrpc": "2.0", "method": "...", "id": <int>, "params": {...}}`.
The `params` field is omitted when a method takes no parameters.

### Method summary

| Method | Description |
|---|---|
| `engine.capabilities` | Version, supported models, devices, profiles |
| `engine.status` | Idle/busy state, loaded models, VRAM usage, uptime |
| `engine.shutdown` | Graceful shutdown |
| `project.scan` | Discover clips in a directory (no job submission) |
| `job.submit` | Submit a generate or inference job |
| `job.cancel` | Cancel the active job |
| `job.status` | Query job progress (for polling clients) |
| `model.status` | Loaded model info and VRAM usage |
| `model.unload` | Explicitly free VRAM |
| `events.subscribe` | Subscribe to event categories |
| `events.unsubscribe` | Unsubscribe from event categories |

---

### `engine.capabilities`

Discover engine version, available generators, backends, devices, and profiles.
Call this first to check compatibility.

```json
// Request
{"jsonrpc": "2.0", "method": "engine.capabilities", "id": 1}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "version": "2.0.0",
    "generators": ["birefnet", "gvm", "videomama"],
    "backends": ["torch", "torch_optimized", "mlx"],
    "devices": [
      {"id": "cuda:0", "name": "RTX 4090", "vram_gb": 24.0}
    ],
    "profiles": ["original", "optimized", "performance", "experimental"],
    "transport": "stdio"
  }
}
```

---

### `engine.status`

Current engine state: idle or busy, which models are loaded, VRAM snapshot.

```json
// Request
{"jsonrpc": "2.0", "method": "engine.status", "id": 2}

// Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "state": "idle",
    "active_job": null,
    "models_loaded": {
      "inference": {
        "backend": "torch_optimized",
        "device": "cuda:0",
        "vram_mb": 1200
      },
      "generator": null
    },
    "vram": {
      "total_mb": 24576,
      "used_mb": 1200,
      "free_mb": 23376
    },
    "uptime_seconds": 342.5
  }
}
```

`state` is one of `"idle"`, `"busy"`, or `"shutting_down"`.

---

### `engine.shutdown`

Graceful shutdown. The engine finishes the current frame (not the current
job), unloads all models, then exits. The response is sent before exit.

```json
// Request
{"jsonrpc": "2.0", "method": "engine.shutdown", "id": 3}

// Response
{"jsonrpc": "2.0", "id": 3, "result": "ok"}
// Process exits shortly after
```

---

### `project.scan`

Scan a directory and return all discovered clips with their asset state.
This is a read-only discovery call — no job is created. Use this to
populate a clip browser without importing any Python modules.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "project.scan",
  "id": 4,
  "params": {
    "path": "/projects/shot01"
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "project_path": "/projects/shot01",
    "is_v2": true,
    "clips": [
      {
        "name": "plate_001",
        "root_path": "/projects/shot01/clips/plate_001",
        "state": "ready",
        "input": {
          "type": "sequence",
          "frame_count": 150,
          "path": "/projects/shot01/clips/plate_001/Input"
        },
        "alpha": {
          "type": "sequence",
          "frame_count": 150,
          "path": "/projects/shot01/clips/plate_001/AlphaHint"
        },
        "mask": null,
        "has_outputs": true,
        "completed_frames": 120
      },
      {
        "name": "plate_002",
        "root_path": "/projects/shot01/clips/plate_002",
        "state": "needs_alpha",
        "input": {
          "type": "sequence",
          "frame_count": 200,
          "path": "/projects/shot01/clips/plate_002/Input"
        },
        "alpha": null,
        "mask": null,
        "has_outputs": false,
        "completed_frames": 0
      }
    ]
  }
}
```

`state` values: `"ready"` (has input and alpha), `"needs_alpha"` (missing
AlphaHint), `"incomplete"` (output frames fewer than input frames),
`"empty"` (no input frames).

---

### `job.submit`

Submit a job for execution. Returns immediately with acceptance or a busy
error. Progress arrives via event notifications.

#### Generate job

Generates alpha matte hints using a neural network. Required before running
inference on new footage.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "job.submit",
  "id": 5,
  "params": {
    "type": "generate",
    "path": "/projects/shot01",
    "model": "birefnet",
    "mode": "fill",
    "frames": null,
    "device": "auto",
    "halt_on_failure": false
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | required | `"generate"` |
| `path` | string | required | Project directory path |
| `model` | string | `"birefnet"` | `"birefnet"`, `"gvm"`, or `"videomama"` |
| `mode` | string | `"fill"` | `"replace"` overwrite all, `"fill"` skip existing, `"skip"` skip clips that have any alpha |
| `frames` | string\|null | `null` | Frame range or null for all. See [Frame Range Syntax](#frame-range-syntax) |
| `device` | string | `"auto"` | `"auto"`, `"cuda:0"`, `"mps"`, `"cpu"` |
| `halt_on_failure` | bool | `false` | Abort job on first frame error |

#### Inference job

Runs the CorridorKey keying model on clips that have both Input and AlphaHint frames.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "job.submit",
  "id": 6,
  "params": {
    "type": "inference",
    "path": "/projects/shot01",
    "frames": null,
    "device": "auto",
    "backend": "auto",
    "settings": {
      "input_is_linear": false,
      "despill_strength": 0.5,
      "auto_despeckle": true,
      "despeckle_size": 400,
      "refiner_scale": 1.0
    },
    "optimization": {
      "profile": "optimized"
    },
    "devices": ["cuda:0"],
    "img_size": 2048,
    "read_workers": 0,
    "write_workers": 0,
    "halt_on_failure": false
  }
}
```

**`settings` fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input_is_linear` | bool | `false` | Input frames are linear light (not sRGB) |
| `despill_strength` | float | `0.0` | Green spill suppression, 0.0–10.0 |
| `auto_despeckle` | bool | `false` | Morphological despeckle (removes tracking markers) |
| `despeckle_size` | int | `400` | Minimum pixel area treated as speckle |
| `refiner_scale` | float | `1.0` | Refiner strength multiplier |

**`optimization` fields** — use `profile` shorthand or set fields individually:

| Field | Type | Description |
|---|---|---|
| `profile` | string | `"optimized"`, `"original"`, `"experimental"`, `"performance"` |
| `flash_attention` | bool | FlashAttention patching (reduces VRAM) |
| `tiled_refiner` | bool | Tile-based CNN refiner (reduces VRAM) |
| `cache_clearing` | bool | CUDA cache clearing between frames |
| `cudnn_benchmark` | bool | cuDNN kernel auto-tune (+2-5 GB VRAM, faster convolutions) |
| `gpu_postprocess` | bool | GPU postprocessing (+~1.5 GB VRAM, faster) |
| `token_routing` | bool | Experimental sparse attention (improves speed at 4K+) |
| `compile_mode` | string | `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `model_precision` | string | `"float16"`, `"bfloat16"`, `"float32"` |
| `tile_size` | int | Tile size for tiled refiner (default 512) |
| `tile_overlap` | int | Tile overlap in pixels (default 128) |
| `comp_format` | string | `"exr"` or `"png"` |

Individual `optimization` fields override the profile defaults. For
example, `{"profile": "optimized", "compile_mode": "reduce-overhead"}`
applies the optimized profile then enables compilation on top.

#### Success response (both job types)

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "job_id": "j-a1b2c3",
    "clips": [
      {"name": "plate_001", "input_frames": 150, "alpha_frames": 150},
      {"name": "plate_002", "input_frames": 200, "alpha_frames": 0}
    ],
    "total_frames": 350
  }
}
```

#### Rejection — engine busy

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32000,
    "message": "Engine busy",
    "data": {"active_job": "j-x9y8z7"}
  }
}
```

---

### `job.cancel`

Cancel the active job. The engine finishes the current frame, then stops.
Emits `event.job.cancelled` when done.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "job.cancel",
  "id": 7,
  "params": {
    "job_id": "j-a1b2c3"
  }
}

// Response
{"jsonrpc": "2.0", "id": 7, "result": "cancelling"}

// Error — wrong job id or no active job
{
  "jsonrpc": "2.0",
  "id": 7,
  "error": {"code": -32001, "message": "Job not found"}
}
```

---

### `job.status`

Query a job's current state. Useful for polling clients or reconnecting
after a disconnect. Prefer events when possible — they carry the same data
without polling overhead.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "job.status",
  "id": 8,
  "params": {
    "job_id": "j-a1b2c3"
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "job_id": "j-a1b2c3",
    "state": "running",
    "type": "inference",
    "current_clip": "plate_001",
    "progress": {"done": 45, "total": 150},
    "clips_completed": 0,
    "clips_total": 2,
    "elapsed_seconds": 12.3
  }
}
```

`state` is one of `"running"`, `"completed"`, `"failed"`, or `"cancelled"`.

---

### `model.status`

What models are currently loaded, which device they are on, and how much
VRAM they occupy.

```json
// Request
{"jsonrpc": "2.0", "method": "model.status", "id": 9}

// Response
{
  "jsonrpc": "2.0",
  "id": 9,
  "result": {
    "inference_engine": {
      "backend": "torch_optimized",
      "device": "cuda:0",
      "img_size": 2048,
      "precision": "float16",
      "vram_mb": 1200,
      "config_hash": "a3f8b91c"
    },
    "generator": null
  }
}
```

`config_hash` is a hash of the `OptimizationConfig` used to load the
model. The engine uses this internally to decide whether to reuse or
reload; it is exposed here for diagnostics.

---

### `model.unload`

Explicitly free VRAM. Use this before running a GPU-intensive task outside
the engine, or to force a reload of the model on the next job.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "model.unload",
  "id": 10,
  "params": {
    "which": "all"
  }
}

// Response
{"jsonrpc": "2.0", "id": 10, "result": {"freed_mb": 1200}}
```

`which` accepts `"all"`, `"inference"`, or `"generator"`.

---

### `events.subscribe`

Subscribe to event categories. Only subscribed events are delivered.
The default on connection is all events (`"all"`).

```json
// Request — subscribe to job and model events, suppress log spam
{
  "jsonrpc": "2.0",
  "method": "events.subscribe",
  "id": 11,
  "params": {
    "categories": ["job", "model"]
  }
}

// Response
{"jsonrpc": "2.0", "id": 11, "result": "ok"}
```

Valid categories: `"job"`, `"model"`, `"log"`, `"warning"`, `"all"`.

---

### `events.unsubscribe`

Remove a subscription. Events in the specified categories stop being delivered.

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "events.unsubscribe",
  "id": 12,
  "params": {
    "categories": ["log"]
  }
}

// Response
{"jsonrpc": "2.0", "id": 12, "result": "ok"}
```

---

## Event Notifications

The engine pushes notifications as JSON-RPC messages with no `id` field.
They arrive on the same stream as responses. A notification looks like:

```json
{"jsonrpc": "2.0", "method": "event.<category>.<name>", "params": {...}}
```

Notifications are only sent for categories the client has subscribed to.

---

### Job events

#### `event.job.accepted`

Sent immediately after `job.submit` succeeds. Confirms the job has been
accepted and provides the total frame count for progress tracking.

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.accepted",
  "params": {
    "job_id": "j-a1b2c3",
    "type": "inference",
    "total_frames": 350
  }
}
```

#### `event.job.clip_started`

Sent when the engine begins processing a clip.

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.clip_started",
  "params": {
    "job_id": "j-a1b2c3",
    "clip": "plate_001",
    "frames": 150,
    "clip_index": 0,
    "clips_total": 2
  }
}
```

#### `event.job.progress`

Sent after each frame completes. High frequency — filter or debounce in
UI code if needed.

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.progress",
  "params": {
    "job_id": "j-a1b2c3",
    "clip": "plate_001",
    "done": 45,
    "total": 150,
    "bytes_read": 123456,
    "bytes_written": 234567,
    "fps": 12.3
  }
}
```

#### `event.job.clip_completed`

Sent when a clip finishes (all frames processed or skipped).

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.clip_completed",
  "params": {
    "job_id": "j-a1b2c3",
    "clip": "plate_001",
    "frames_ok": 148,
    "frames_failed": 2
  }
}
```

#### `event.job.completed`

Sent when the entire job finishes successfully. `failed_frames` lists any
individual frame errors even if the job completed overall.

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.completed",
  "params": {
    "job_id": "j-a1b2c3",
    "clips_ok": 2,
    "clips_failed": 0,
    "total_frames": 350,
    "frames_ok": 348,
    "frames_failed": 2,
    "elapsed_seconds": 45.2,
    "failed_frames": [
      {"clip": "plate_001", "frame": 42, "error": "OOM"}
    ]
  }
}
```

#### `event.job.failed`

Sent when the job aborts. No `event.job.completed` follows.

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.failed",
  "params": {
    "job_id": "j-a1b2c3",
    "error": "No clips with input frames found"
  }
}
```

#### `event.job.cancelled`

Sent after `job.cancel` takes effect (current frame finished).

```json
{
  "jsonrpc": "2.0",
  "method": "event.job.cancelled",
  "params": {
    "job_id": "j-a1b2c3",
    "frames_completed": 45
  }
}
```

---

### Model events

#### `event.model.loading`

Sent when a model starts loading into VRAM. Expect a delay of several
seconds before `event.model.loaded` arrives.

```json
{
  "jsonrpc": "2.0",
  "method": "event.model.loading",
  "params": {
    "model": "birefnet",
    "device": "cuda:0"
  }
}
```

#### `event.model.loaded`

Sent when a model is ready to use.

```json
{
  "jsonrpc": "2.0",
  "method": "event.model.loaded",
  "params": {
    "model": "birefnet",
    "device": "cuda:0",
    "vram_mb": 3500,
    "load_seconds": 8.2
  }
}
```

#### `event.model.unloaded`

Sent when a model is unloaded from VRAM (explicit `model.unload` call,
config change, or shutdown).

```json
{
  "jsonrpc": "2.0",
  "method": "event.model.unloaded",
  "params": {
    "model": "inference",
    "freed_mb": 1200
  }
}
```

#### `event.model.recompiling`

Sent when the optimization config changed between jobs and the engine must
reload and recompile the model. Expect a longer first-frame warmup.

```json
{
  "jsonrpc": "2.0",
  "method": "event.model.recompiling",
  "params": {
    "reason": "optimization config changed",
    "backend": "torch_optimized"
  }
}
```

---

### Log events

Diagnostic messages from the Python logging system, structured and opt-in.
Subscribe to `"log"` to receive them. Do not parse stderr for this data.

```json
{
  "jsonrpc": "2.0",
  "method": "event.log",
  "params": {
    "level": "info",
    "message": "Inference settings: linear=False, despill=0.5, despeckle=True",
    "logger": "ck_engine.pipeline.inference",
    "timestamp": 1710547200.123
  }
}

{
  "jsonrpc": "2.0",
  "method": "event.log",
  "params": {
    "level": "warning",
    "message": "Frame count mismatch: input=150, alpha=148",
    "logger": "ck_engine.validators",
    "timestamp": 1710547201.456
  }
}
```

`level` is one of `"debug"`, `"info"`, `"warning"`, `"error"`.

---

## Error Codes

Standard JSON-RPC errors (-32768 to -32000) plus application-specific codes.

| Code | Name | Meaning |
|---|---|---|
| -32700 | Parse error | Malformed JSON in request body |
| -32600 | Invalid request | Not a valid JSON-RPC 2.0 object |
| -32601 | Method not found | Unknown method name |
| -32602 | Invalid params | Missing or wrong-type params |
| -32000 | Engine busy | A job is already running — only one at a time |
| -32001 | Job not found | Requested `job_id` does not exist |
| -32002 | Invalid path | Path is not a valid project or clip directory |
| -32003 | No valid clips | No clips with input frames found at path |
| -32004 | Model load failure | Model weights missing or device OOM |
| -32005 | Device unavailable | Requested device not present or not supported |
| -32006 | Cancelled | Operation was cancelled |

Error responses always include `code` and `message`. Application errors
(-32000 to -32006) include a `data` field with structured details where
available:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32000,
    "message": "Engine busy",
    "data": {
      "active_job": "j-x9y8z7"
    }
  }
}
```

---

## Frame Range Syntax

The `frames` field in `job.submit` accepts a string using a compact range
syntax, or `null` to process all frames.

| Value | Meaning |
|---|---|
| `null` | All frames |
| `"1-100"` | Frames 1 through 100, inclusive (1-based) |
| `"1,5,10-20"` | Frames 1, 5, and 10 through 20 |
| `"50-"` | Frame 50 to the last frame |

Frames are 1-based. Out-of-range frames are silently ignored (they will
not appear in job output, but the job does not error).

Parsed by `ck_engine.api.frames.parse_frame_range(spec: str, total: int) -> list[int]`.

Examples:

```python
from ck_engine.api.frames import parse_frame_range

parse_frame_range("1-10", 20)      # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parse_frame_range("1,5,10-20", 30) # [1, 5, 10, 11, 12, ..., 20]
parse_frame_range("50-", 75)       # [50, 51, ..., 75]
parse_frame_range(None, 100)       # [1, 2, ..., 100]
```

---

## Model Caching

The engine keeps models resident in VRAM between jobs to avoid repeated
load times.

**Reuse conditions:** If consecutive jobs use the same optimization config,
the same model backend, the same device, and the same `img_size`, the
loaded model is reused with no delay.

**Reload conditions:** If any of those parameters differ, the engine:

1. Unloads the old model and frees its VRAM.
2. Calls `torch._dynamo.reset()` to clear `torch.compile` state.
3. Loads a fresh model with the new config.
4. Emits `event.model.recompiling` before reload and `event.model.loaded` when ready.

**Multi-GPU:** Each device tracks its own loaded model independently. A
job running on `cuda:0` and `cuda:1` loads two model copies; each is
reused or reloaded independently.

**Explicit unload:** Call `model.unload` to free VRAM on demand — for
example, before handing off the GPU to a compositor or another process.

**Caching is never a correctness requirement.** Each job carries its full
configuration. Killing and restarting the engine produces identical output.

---

## Subscription Model

The engine uses an opt-in pub/sub system for event notifications.

**Default on connect:** All events (`"all"`) are delivered. This matches
the behaviour most clients want without any setup call.

**Subscribe to specific categories:**

```json
{"jsonrpc": "2.0", "method": "events.subscribe", "id": 1,
 "params": {"categories": ["job", "model"]}}
```

Calling `events.subscribe` replaces the current subscription for the
specified categories (it does not add to them). Calling
`events.unsubscribe` removes those categories.

**Categories:**

| Category | Events included |
|---|---|
| `"job"` | `event.job.*` — accepted, clip_started, progress, clip_completed, completed, failed, cancelled |
| `"model"` | `event.model.*` — loading, loaded, unloaded, recompiling |
| `"log"` | `event.log` — diagnostic messages at all levels |
| `"warning"` | `event.log` messages at `warning` or `error` level only |
| `"all"` | All of the above |

**Recommended patterns:**

- **Progress UI** (Blender plugin, desktop app): subscribe to `["job", "model"]`. Suppress log noise.
- **Headless scripting**: subscribe to `["job"]`. Only care about completion.
- **Debug / developer**: subscribe to `["all"]`. Full visibility.
- **Log aggregator**: subscribe to `["log"]` only.

Subscribe once after connecting. The subscription persists for the
lifetime of the connection; there is no per-job subscription.
