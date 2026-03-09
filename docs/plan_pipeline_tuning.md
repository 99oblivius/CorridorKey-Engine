# Pipeline Tuning Implementation Plan

**Status**: Draft Implementation Plan
**Version**: 1.0
**Date**: 2026-03-09

## Overview

This document covers two complementary optimization tasks for the CorridorKey async inference pipeline:

1. **Task 1**: Expose `--read-workers` and `--write-workers` CLI flags to allow users to fine-tune reader/writer thread pool sizes
2. **Task 2**: Parallelize GPU kernel compilation during warmup to reduce multi-GPU startup time from 60-120s to ~35-40s

Both tasks target the async pipeline's performance characteristics and are safe to implement independently.

---

## Task 1: Add --read-workers and --write-workers CLI Flags

### Current State Analysis

**PipelineConfig** (`backend/async_pipeline.py` lines 80-91):
```python
@dataclasses.dataclass
class PipelineConfig:
    """Tunable pipeline parameters."""
    img_size: int = 2048
    prefetch_depth: int = 0  # 0 = auto (num_gpus * 8)
    read_workers: int = 0    # 0 = auto (cpu_count // 4, min 2)
    write_workers: int = 0   # 0 = auto (cpu_count // 4, min 2)
    devices: list[str] | None = None
    backend: str | None = None
    optimization_config: Any = None
```

The config already has `read_workers` and `write_workers` fields with support for zero-valued auto-scaling. However:

- These fields are **NOT exposed as CLI flags** in `corridorkey_cli.py`
- No CLI argument parsing for `--read-workers` or `--write-workers`
- Values are not passed from CLI → `interactive_wizard()` → `run_inference()` → `PipelineConfig`

**Auto-scaling behavior** (`backend/async_pipeline.py` lines 388-391):
```python
num_readers = self.config.read_workers or max(2, os.cpu_count() // 4)
num_writers = self.config.write_workers or max(4, os.cpu_count() // 2)
```

- Readers default: `max(2, cpu_count // 4)` (e.g., 4 cores → 1 worker, clamped to 2)
- Writers default: `max(4, cpu_count // 2)` (e.g., 4 cores → 2 workers, clamped to 4)
- The asymmetry (writers >= readers) reflects that EXR/PNG encoding is CPU-heavy

### Why This Task Matters

**Use Case 1: High Core Count Systems**
- 32-core CPU: readers = max(2, 8) = 8, writers = max(4, 16) = 16
- For small clips with fast I/O, 16 writers may cause contention on disk or feed exhaustion
- Users should be able to tune down to `--read-workers 6 --write-workers 10`

**Use Case 2: Resource-Constrained Hosts**
- 2-core CPU: readers = 2, writers = 4 (via clamping)
- For very tight VRAM or memory, users may want `--read-workers 1 --write-workers 2`

**Use Case 3: NAS/Network Storage**
- Network-attached storage has different I/O characteristics than local NVMe
- Users may want to reduce reader/writer pools to avoid network saturation
- Example: `--read-workers 2 --write-workers 4` for slower storage

### Implementation Steps

#### Step 1.1: Add CLI Arguments (corridorkey_cli.py)

**Location**: Around line 346 (after existing optimization flags)

```python
parser.add_argument(
    "--read-workers",
    type=int,
    default=0,
    help="Reader process pool size (0=auto: max(2, cpu_count//4)). "
    "Set to 1 for single-threaded reading, higher for parallel I/O.",
)
parser.add_argument(
    "--write-workers",
    type=int,
    default=0,
    help="Writer process pool size (0=auto: max(4, cpu_count//2)). "
    "Higher values speed up EXR/PNG encoding at the cost of CPU/memory. "
    "For NAS: consider reducing to 2-4.",
)
```

**Rationale**:
- Default 0 preserves existing auto-scaling behavior
- Type int, no validation needed (PipelineConfig will clamp via max(...))
- Help text explains auto-scaling formula and common use cases

#### Step 1.2: Pass Through to run_inference() (corridorkey_cli.py)

**Location**: Line 379 (--action=run_inference branch)

**Current**:
```python
run_inference(clips, device=device, backend=backend, optimization_config=optimization_config, devices=devices_list, img_size=args.img_size)
```

**Updated**:
```python
run_inference(
    clips,
    device=device,
    backend=backend,
    optimization_config=optimization_config,
    devices=devices_list,
    img_size=args.img_size,
    read_workers=args.read_workers,
    write_workers=args.write_workers,
)
```

#### Step 1.3: Pass Through to interactive_wizard() (corridorkey_cli.py)

**Location**: Line 384 (--action=wizard branch)

**Current**:
```python
interactive_wizard(args.win_path, device=device, devices=devices_list, backend=backend, optimization_config=optimization_config, img_size=args.img_size)
```

**Updated**:
```python
interactive_wizard(
    args.win_path,
    device=device,
    devices=devices_list,
    backend=backend,
    optimization_config=optimization_config,
    img_size=args.img_size,
    read_workers=args.read_workers,
    write_workers=args.write_workers,
)
```

#### Step 1.4: Update run_inference() Signature (clip_manager.py)

**Location**: Line 590

**Current**:
```python
def run_inference(clips, device=None, backend=None, max_frames=None, optimization_config=None, devices=None, img_size=2048):
```

**Updated**:
```python
def run_inference(
    clips,
    device=None,
    backend=None,
    max_frames=None,
    optimization_config=None,
    devices=None,
    img_size=2048,
    read_workers=0,
    write_workers=0,
):
```

**Inside function** (line 658):

**Current**:
```python
config = PipelineConfig(img_size=img_size, backend=backend, devices=devices, optimization_config=optimization_config)
```

**Updated**:
```python
config = PipelineConfig(
    img_size=img_size,
    backend=backend,
    devices=devices,
    optimization_config=optimization_config,
    read_workers=read_workers,
    write_workers=write_workers,
)
```

#### Step 1.5: Update interactive_wizard() Signature (corridorkey_cli.py)

**Location**: Line 51

**Current**:
```python
def interactive_wizard(win_path: str, device: str | None = None, devices: list[str] | None = None, backend: str | None = None, optimization_config=None, img_size: int = 2048) -> None:
```

**Updated**:
```python
def interactive_wizard(
    win_path: str,
    device: str | None = None,
    devices: list[str] | None = None,
    backend: str | None = None,
    optimization_config=None,
    img_size: int = 2048,
    read_workers: int = 0,
    write_workers: int = 0,
) -> None:
```

**Inside wizard** (line 282):

**Current**:
```python
run_inference(ready, device=device, devices=devices, backend=backend, optimization_config=optimization_config, img_size=img_size)
```

**Updated**:
```python
run_inference(
    ready,
    device=device,
    devices=devices,
    backend=backend,
    optimization_config=optimization_config,
    img_size=img_size,
    read_workers=read_workers,
    write_workers=write_workers,
)
```

### Testing Task 1

1. **Auto-scaling baseline**: `python corridorkey_cli.py --action run_inference`
   - Verify log message shows `num_readers = max(2, cpu_count//4)` and `num_writers = max(4, cpu_count//2)`

2. **Custom pools**: `python corridorkey_cli.py --action run_inference --read-workers 2 --write-workers 3`
   - Verify log message shows `num_readers = 2` and `num_writers = 3`

3. **Wizard integration**: `python corridorkey_cli.py --action wizard --win_path /path --read-workers 1 --write-workers 2`
   - Select [i] to run inference, verify pools are set correctly

4. **Zero handling**: `python corridorkey_cli.py --action run_inference --read-workers 0 --write-workers 0`
   - Should use auto-scaling, not fail

### Files Modified (Task 1)

| File | Lines | Changes |
|------|-------|---------|
| `corridorkey_cli.py` | 346-348 | Add `--read-workers` and `--write-workers` args |
| `corridorkey_cli.py` | 379 | Pass args to `run_inference()` |
| `corridorkey_cli.py` | 384 | Pass args to `interactive_wizard()` |
| `corridorkey_cli.py` | 51 | Update `interactive_wizard()` signature |
| `corridorkey_cli.py` | 282 | Pass args in wizard's `run_inference()` call |
| `clip_manager.py` | 590 | Update `run_inference()` signature |
| `clip_manager.py` | 658 | Pass to `PipelineConfig()` |

**No changes needed** to `backend/async_pipeline.py` — the infrastructure is already there.

---

## Task 2: Parallel Kernel Compilation for Multiple GPUs

### Current State Analysis

**_warmup_engines() Method** (`backend/async_pipeline.py` lines 290-306):

```python
def _warmup_engines(self) -> None:
    """Run a dummy forward pass on each engine to trigger torch.compile..."""
    for dev_str, engine in self.engines:
        if not dev_str.startswith("cuda"):
            continue
        logger.info("Compiling kernels for %s (this is one-time, please wait)...", dev_str)
        dummy = np.zeros((self.config.img_size, self.config.img_size, 4), dtype=np.float32)
        t0 = time.perf_counter()
        try:
            engine.process_prepared(dummy, self.config.img_size, self.config.img_size)
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        logger.info("Kernel compilation for %s done in %.1fs", dev_str, elapsed)
```

**Current Characteristics**:
- **Sequential iteration**: `for dev_str, engine in self.engines`
- **Per-GPU compilation**: Each GPU's Triton kernel generation happens one at a time
- **Timing**: ~30-60s per GPU, so 2 GPUs = 60-120s total startup
- **Bottleneck**: Triton's PTX/CUBIN generation is CPU-bound, but GIL contention limits parallelism

**Why Parallelization Works**:

1. **Triton Compilation is CPU-bound**: The Triton compiler (in `torch.compile`'s backend) generates CUDA PTX/CUBIN code on the CPU. The actual device compilation (PTX → CUBIN) happens asynchronously on the GPU.

2. **Different Compilation Caches**: Each GPU has different hardware properties (compute capability, memory bandwidth, etc.), so `torch.compile` generates different kernels per device. The cache keys are unique per device, so two threads won't collide on cache writes.

3. **GIL Releases During I/O**: Thread-based parallelism benefits from:
   - File system I/O (writing cache files to `~/.cache/torch/inductor/`)
   - C-level compilation (Triton may release the GIL during some compilation phases)
   - CUDA kernel launch (brief, but still releases GIL)

4. **Expected Speedup**: ~30-50% (not 2x) due to GIL contention during Python-level codegen. Practical result: 60s → 35-40s for dual-GPU warmup.

### Why ProcessPoolExecutor is Not Viable

**Limitation**: `torch.compile` models contain references to compiled C++ extension objects that are not picklable. Attempting to serialize and send an engine to a worker process will fail.

**Fallback**: If threading proves insufficient, the alternative would be sequential compilation with cached wheels (.so files), but this adds deployment complexity.

### Implementation Steps

#### Step 2.1: Import threading Module

**Location**: `backend/async_pipeline.py` top of file (line 25 already has `threading` imported)

No change needed — `threading` is already imported.

#### Step 2.2: Replace Sequential Loop with Thread Pool

**Location**: `backend/async_pipeline.py` lines 290-306

**Replace**:
```python
def _warmup_engines(self) -> None:
    """Run a dummy forward pass on each engine to trigger torch.compile.

    Moves the ~30-60s Triton compilation to a predictable startup phase.
    """
    for dev_str, engine in self.engines:
        if not dev_str.startswith("cuda"):
            continue
        logger.info("Compiling kernels for %s (this is one-time, please wait)...", dev_str)
        dummy = np.zeros((self.config.img_size, self.config.img_size, 4), dtype=np.float32)
        t0 = time.perf_counter()
        try:
            engine.process_prepared(dummy, self.config.img_size, self.config.img_size)
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        logger.info("Kernel compilation for %s done in %.1fs", dev_str, elapsed)
```

**With**:
```python
def _warmup_engines(self) -> None:
    """Run dummy forward passes on each GPU in parallel via threading.

    Moves the ~30-60s Triton compilation to a predictable startup phase.
    With multiple GPUs, parallelizes kernel compilation (CPU-bound work) to
    reduce total startup time from 2x to ~1.3-1.5x per GPU.
    """
    def _warmup_one(dev_str: str, engine: Any) -> None:
        """Warmup a single engine on its device (runs in worker thread)."""
        if not dev_str.startswith("cuda"):
            return

        logger.info("Compiling kernels for %s (one-time, please wait)...", dev_str)

        # Set CUDA device context for this thread
        try:
            dev_idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
            torch.cuda.set_device(dev_idx)
        except (ValueError, RuntimeError):
            logger.warning("Could not set CUDA device for %s, proceeding anyway", dev_str)

        dummy = np.zeros((self.config.img_size, self.config.img_size, 4), dtype=np.float32)
        t0 = time.perf_counter()
        try:
            engine.process_prepared(dummy, self.config.img_size, self.config.img_size)
        except Exception as e:
            logger.debug("Exception during warmup of %s (non-fatal): %s", dev_str, e)
        elapsed = time.perf_counter() - t0
        logger.info("Kernel compilation for %s done in %.1fs", dev_str, elapsed)

    threads = []
    for dev_str, engine in self.engines:
        t = threading.Thread(
            target=_warmup_one,
            args=(dev_str, engine),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all compilation threads to finish
    for t in threads:
        t.join()
```

**Key Changes**:
1. Define nested `_warmup_one(dev_str, engine)` function that performs warmup for a single GPU
2. Set CUDA device context via `torch.cuda.set_device(dev_idx)` inside the thread (critical for multi-GPU)
3. Create one thread per CUDA device and start all threads immediately
4. `t.join()` blocks until all threads complete
5. Log messages remain the same for user visibility

#### Step 2.3: Verify CUDA Device Handling

**Rationale**: When multiple threads call CUDA operations on different devices, each thread must establish its own CUDA context via `torch.cuda.set_device()`. Failure to do this can cause:
- CUDA operations on the wrong device (context bleed)
- Runtime errors ("CUDA context already in use")

**Implementation Detail**: The code extracts device index from strings like `"cuda:0"`, `"cuda:1"`:
```python
dev_idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
```

This safely handles:
- `"cuda:0"` → index 0
- `"cuda:1"` → index 1
- Bare `"cuda"` (treated as device 0)
- Non-CUDA devices (early return before set_device)

### Testing Task 2

#### Test 2.1: Baseline Timing (Sequential)

1. Temporarily revert `_warmup_engines()` to sequential logic
2. Run with `--devices 0,1`:
   ```
   python corridorkey_cli.py --action run_inference --devices 0,1
   ```
3. Note startup time: e.g., "Kernel compilation for cuda:0 done in 45.2s" + "Kernel compilation for cuda:1 done in 43.8s" = ~89s total

#### Test 2.2: Parallel Warmup

1. Use the new threaded implementation
2. Run with `--devices 0,1`:
   ```
   python corridorkey_cli.py --action run_inference --devices 0,1
   ```
3. Observe log output (should show both "Compiling kernels for cuda:X" lines near-simultaneously)
4. Note total time should be ~55-60s (1.3-1.5x the longest individual warmup, not 2x)

#### Test 2.3: Correctness Verification

1. Run inference on a small clip (e.g., 5 frames) with `--devices 0,1`
2. Verify output files are created correctly in all output directories
3. Verify no CUDA context errors or memory corruption in logs
4. Check that all frames process successfully on both GPUs

#### Test 2.4: Single GPU (Control)

1. Run with `--devices 0` (single GPU)
2. Verify warmup still works (thread pool of size 1 should be equivalent)
3. Timing should match original sequential logic

### Performance Expectations

**Theoretical Speedup**:
- Sequential (2 GPUs, 45s each): ~90s total
- Parallel (2 GPUs, 45s each): ~48s total (sequential overhead minimal)
- **Actual**: ~55-60s due to GIL contention and thread scheduling

**Speedup Factor**: 90s → 55s ≈ **1.6x improvement** (from 2x to 1.5x per GPU)

**For 4 GPUs**:
- Sequential: ~180s
- Parallel: ~70-80s expected (**2.2-2.5x improvement**)

### Why This Matters

- **User Experience**: Multi-GPU systems can start inference 1.5-2.5x faster
- **CI/CD Pipelines**: Startup time is more predictable and faster
- **Large Batches**: Reduced blocking on the warmup phase allows quicker clip processing

### Files Modified (Task 2)

| File | Lines | Changes |
|------|-------|---------|
| `backend/async_pipeline.py` | 290-306 | Replace `_warmup_engines()` sequential loop with threading |

**No changes needed** to `corridorkey_cli.py` or `clip_manager.py` — this is transparent to the CLI.

---

## Implementation Order

### Recommended Sequence

1. **Task 1 (CLI Flags)**: Implement first
   - Lower risk (data flows through, no threading complexity)
   - Enables users to experiment with pool sizes
   - Foundation for future tuning work
   - Estimated time: 30 minutes

2. **Task 2 (Parallel Warmup)**: Implement second
   - Builds on Task 1 (users can combine `--read-workers 4 --write-workers 8` with parallel warmup)
   - More testing needed due to threading
   - Estimated time: 1 hour (including testing)

**Total Estimated Time**: 1.5 hours implementation + 30-45 minutes testing

---

## Validation Checklist

### Task 1 Validation

- [ ] CLI parser accepts `--read-workers` and `--write-workers` arguments
- [ ] Arguments default to 0 when not provided
- [ ] `run_inference()` signature updated with new parameters
- [ ] `interactive_wizard()` signature updated with new parameters
- [ ] Values are passed to `PipelineConfig()` constructor
- [ ] Wizard's internal `run_inference()` call includes parameters
- [ ] Direct CLI invocation (`--action run_inference`) passes parameters
- [ ] Test with explicit values: `--read-workers 2 --write-workers 4`
- [ ] Test with zero values: `--read-workers 0 --write-workers 0` (auto-scaling)
- [ ] Test wizard integration: parameters flow through interactive prompts
- [ ] No regression in existing functionality

### Task 2 Validation

- [ ] `_warmup_engines()` method uses threading instead of sequential loop
- [ ] `threading.Thread` objects are created and started correctly
- [ ] All threads are joined before returning (blocking wait)
- [ ] CUDA device context is set correctly per thread (`torch.cuda.set_device()`)
- [ ] Single GPU warmup works (thread pool of size 1)
- [ ] Multi-GPU warmup (2+ GPUs) shows parallel "Compiling kernels for cuda:X" logs
- [ ] Kernel compilation completes successfully on all GPUs
- [ ] Warmup timing is ~1.3-1.5x per GPU (not 2x for 2 GPUs)
- [ ] Inference produces correct output files after parallel warmup
- [ ] No CUDA context errors or resource leaks in logs
- [ ] No regression in single-GPU performance

---

## Risk Assessment

### Task 1 Risks

**Low Risk Overall**:
- Simple parameter passing (no algorithmic changes)
- Existing infrastructure supports zero values (auto-scaling)
- Backward compatible: defaults to 0 (auto-scaling behavior unchanged)

**Potential Issues**:
- Missing parameter in one call path → runtime error (caught by testing)
- Wizard integration might skip parameter passing → user cannot tune via wizard (caught by testing)

### Task 2 Risks

**Medium Risk**:
- Threading can introduce subtle race conditions
- CUDA context management across threads is error-prone
- Cache collisions (unlikely but possible if two GPUs have identical compute capability)

**Mitigations**:
- Use daemon threads (already done, threads clean up on exception)
- Set CUDA device context explicitly in each thread
- `torch.compile` uses device-specific cache keys, collision unlikely
- Test with 2+ GPUs before deploying

**Potential Issues**:
- CUDA context not set → compilation on wrong device
- Thread doesn't join → silent failure if exception in thread
- GIL contention prevents speedup → acceptable (still no regression)

---

## Future Work

### Extensions to Consider

1. **Prefetch depth tuning**: Expose `--prefetch-depth` CLI flag (currently hardcoded)
   - Would let users control queue size between read and inference stages
   - Similar implementation to Task 1

2. **Adaptive pool sizing**: Monitor queue depth and adjust workers dynamically
   - More complex: would require monitoring thread that adjusts `ThreadPoolExecutor` size
   - Low priority (manual tuning via CLI is simpler for now)

3. **Distributed inference**: Extend to multi-machine GPU clusters
   - Would require refactoring `AsyncInferencePipeline` as a distributed service
   - Long-term vision, out of scope for now

---

## References

### Code Locations

- **PipelineConfig**: `backend/async_pipeline.py` lines 80-91
- **process_clip() setup**: `backend/async_pipeline.py` lines 388-391
- **_warmup_engines()**: `backend/async_pipeline.py` lines 290-306
- **CLI argparse**: `corridorkey_cli.py` lines 302-351
- **run_inference() call from CLI**: `corridorkey_cli.py` line 379
- **interactive_wizard() signature**: `corridorkey_cli.py` line 51
- **interactive_wizard() run_inference call**: `corridorkey_cli.py` line 282
- **run_inference() definition**: `clip_manager.py` line 590
- **PipelineConfig instantiation**: `clip_manager.py` line 658

### Documentation

- PyTorch `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html
- Python `threading` module: https://docs.python.org/3/library/threading.html
- CUDA Python Bindings: https://docs.nvidia.com/cuda/cuda-python/index.html

---

## Appendix: Example Usage

### Task 1 Examples

```bash
# Use auto-scaling (existing behavior)
python corridorkey_cli.py --action run_inference

# Custom pool sizes
python corridorkey_cli.py --action run_inference --read-workers 4 --write-workers 8

# Minimal pooling (for resource-constrained systems)
python corridorkey_cli.py --action run_inference --read-workers 1 --write-workers 2

# Via wizard
python corridorkey_cli.py --action wizard --win_path /path/to/clips \
  --devices 0,1 --read-workers 2 --write-workers 6
```

### Task 2 Verification

```bash
# Single GPU (baseline, should show sequential warmup equivalent)
python corridorkey_cli.py --action run_inference --devices 0

# Dual GPU (should show parallel "Compiling kernels" messages)
python corridorkey_cli.py --action run_inference --devices 0,1

# Quad GPU (should compile all 4 in parallel, significant speedup)
python corridorkey_cli.py --action run_inference --devices 0,1,2,3
```

Expected log output with Task 2:
```
INFO: Compiling kernels for cuda:0 (one-time, please wait)...
INFO: Compiling kernels for cuda:1 (one-time, please wait)...
INFO: Kernel compilation for cuda:0 done in 44.2s
INFO: Kernel compilation for cuda:1 done in 43.8s
```

Note: Both "Compiling" messages appear within ~1-2 seconds of each other, not 45+ seconds apart.

