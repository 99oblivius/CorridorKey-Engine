"""Async pipelined inference engine for CorridorKey.

Overlaps frame I/O with GPU inference using a multi-stage concurrent
pipeline. Supports multiple GPUs via work-stealing.

Architecture:
    Read Workers (ThreadPool) → Work Queue → Inference Threads (1:1 per GPU)
    → Write Pool (ProcessPool) — parallel disk I/O, escapes GIL

Usage:
    from backend.async_pipeline import AsyncInferencePipeline, PipelineConfig
    config = PipelineConfig(img_size=2048)
    pipeline = AsyncInferencePipeline(config)
    pipeline.process_clip(clip, settings)
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Sentinel to signal shutdown
_SHUTDOWN = object()


class _TimelineProfiler:
    """Thread-safe timeline profiler that records span events across all workers.

    Each event is a (timestamp, thread/device, stage, start/end, frame_index)
    tuple.  At the end of a run, produces a console summary with p50 fps,
    phase durations, and per-GPU stats.
    """

    def __init__(self, t0: float) -> None:
        self._t0 = t0  # pipeline epoch — all timestamps relative to this
        self._events: list[tuple[float, str, str, str, int]] = []
        self._lock = threading.Lock()
        # Per-frame completion timestamps for fps calculation
        self._frame_done_times: list[float] = []

    def span_begin(self, worker: str, stage: str, frame_idx: int = -1) -> float:
        """Record the start of a span.  Returns the timestamp for pairing."""
        t = time.perf_counter()
        with self._lock:
            self._events.append((t - self._t0, worker, stage, "begin", frame_idx))
        return t

    def span_end(self, worker: str, stage: str, t_begin: float, frame_idx: int = -1) -> None:
        """Record the end of a span."""
        t = time.perf_counter()
        with self._lock:
            self._events.append((t - self._t0, worker, stage, "end", frame_idx))

    def mark(self, worker: str, event_name: str, frame_idx: int = -1) -> None:
        """Record a point event (no duration)."""
        t = time.perf_counter()
        with self._lock:
            self._events.append((t - self._t0, worker, event_name, "mark", frame_idx))

    def frame_completed(self) -> None:
        """Record that a frame has been fully written to disk."""
        with self._lock:
            self._frame_done_times.append(time.perf_counter())

    def console_summary(
        self,
        num_frames: int,
        total_wall: float,
        warmup_time: float,
        read_wall: float,
        infer_wall: float,
        write_drain: float,
        completed_at_infer_done: int = 0,
    ) -> str:
        """Produce concise console summary."""
        with self._lock:
            done_times = sorted(self._frame_done_times)

        # P50 fps: median inter-frame interval during steady-state processing
        fps_str = "n/a"
        if len(done_times) > 2:
            intervals = [done_times[i + 1] - done_times[i] for i in range(len(done_times) - 1)]
            intervals.sort()
            # Take upper 50th percentile (slower half) for conservative estimate
            mid = len(intervals) // 2
            upper_half = intervals[mid:]
            if upper_half:
                median_interval = upper_half[len(upper_half) // 2]
                fps_str = f"{1.0 / median_interval:.2f}"

        lines = [
            "",
            f"  Warmup + compile : {warmup_time:7.2f}s",
            f"  Read phase       : {read_wall:7.2f}s",
            f"  Inference phase  : {infer_wall:7.2f}s",
            f"  Write drain      : {write_drain:7.2f}s",
            f"  Total            : {total_wall:7.2f}s",
            f"  Frames           : {num_frames}",
            f"  Throughput       : {num_frames / total_wall:.2f} fps (overall)  |  {fps_str} fps (p50 sustained)",
            (
                f"  GPU vs CPU       : GPU done at {completed_at_infer_done}/{num_frames}"
                f" written ({completed_at_infer_done * 100 // num_frames}%)"
            ),
        ]

        # Per-GPU stats from events
        gpu_stats = self._per_worker_stats()
        for worker, stats in sorted(gpu_stats.items()):
            if not worker.startswith("gpu:"):
                continue
            forward_avg = stats.get("forward", (0, 0))
            queue_avg = stats.get("queue_wait", (0, 0))
            n_frames = max(forward_avg[1], 1)
            lines.append(
                f"  {worker:14s}  : {n_frames} frames | "
                f"forward {forward_avg[0] / n_frames * 1000:.0f}ms avg | "
                f"idle {queue_avg[0] / max(queue_avg[1], 1) * 1000:.0f}ms avg"
            )

        return "\n".join(lines)

    def _per_worker_stats(self) -> dict[str, dict[str, tuple[float, int]]]:
        """Aggregate total time and count per (worker, stage) from span pairs."""
        with self._lock:
            events = list(self._events)

        # Match begin/end pairs
        open_spans: dict[tuple[str, str, int], float] = {}  # (worker, stage, frame) -> begin_time
        stats: dict[str, dict[str, tuple[float, int]]] = {}  # worker -> stage -> (total_s, count)

        for t, worker, stage, typ, frame_idx in events:
            if typ == "begin":
                open_spans[(worker, stage, frame_idx)] = t
            elif typ == "end":
                key = (worker, stage, frame_idx)
                if key in open_spans:
                    elapsed = t - open_spans.pop(key)
                    if worker not in stats:
                        stats[worker] = {}
                    prev = stats[worker].get(stage, (0.0, 0))
                    stats[worker][stage] = (prev[0] + elapsed, prev[1] + 1)

        return stats


@dataclasses.dataclass
class PipelineConfig:
    """Tunable pipeline parameters."""

    img_size: int = 2048
    prefetch_depth: int = 0  # 0 = auto (num_gpus * 8)
    read_workers: int = 0  # 0 = auto (cpu_count // 4, min 2)
    write_workers: int = 0  # 0 = auto (cpu_count // 4, min 2)
    devices: list[str] | None = None  # None = auto-detect
    backend: str | None = None  # "torch", "mlx", or None for auto
    optimization_config: Any = None  # OptimizationConfig or None for engine defaults
    output_comp_png: bool = True  # Whether to write comp PNG files


@dataclasses.dataclass
class InferenceSettings:
    """Per-run settings from user prompts."""

    input_is_linear: bool = False
    despill_strength: float = 0.5
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0


@dataclasses.dataclass
class FramePacket:
    """Work item flowing through the pipeline.

    Carries raw decoded frames at original resolution. Resize, color space
    conversion, and normalization are done on GPU by ``process_raw()``,
    reducing IPC transfer size and eliminating CPU-bound preprocessing.
    """

    index: int  # Original frame order
    input_stem: str  # Output filename stem
    img_raw: np.ndarray  # Decoded input (orig_h, orig_w, 3) float32 0-1
    mask_raw: np.ndarray  # Decoded alpha hint (orig_h, orig_w) float32 0-1
    orig_h: int  # Original image height
    orig_w: int  # Original image width


@dataclasses.dataclass
class ResultPacket:
    """Inference output ready for writing."""

    index: int
    input_stem: str
    alpha: np.ndarray
    fg: np.ndarray
    comp: np.ndarray
    processed: np.ndarray | None


def _read_input_frame(
    path: str,
    is_exr: bool,
) -> np.ndarray | None:
    """Read and normalize a single input frame to float32 (H,W,3) 0-1."""
    if is_exr:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def _read_alpha_frame(path: str) -> np.ndarray | None:
    """Read and normalize a single alpha frame to float32 (H,W) 0-1."""
    is_exr = path.lower().endswith(".exr")
    if is_exr:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        mask = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    if mask.dtype == np.uint16:
        return mask.astype(np.float32) / 65535.0
    return mask.astype(np.float32)


def _read_frame_pair(
    input_path: str,
    alpha_path: str,
    index: int,
    input_stem: str,
) -> FramePacket | None:
    """Read one input+alpha pair and return a FramePacket with raw decoded data.

    Only decodes and normalizes to float32 0-1.  Resize, color space
    conversion, and ImageNet normalization are done on GPU by the engine's
    ``process_raw()`` method.  This keeps the IPC payload small (raw
    resolution vs 2048² preprocessed) and frees reader CPU cores.
    """
    is_exr = input_path.lower().endswith(".exr")
    img = _read_input_frame(input_path, is_exr)
    if img is None:
        logger.warning("Failed to read input frame: %s", input_path)
        return None

    mask = _read_alpha_frame(alpha_path)
    if mask is None:
        logger.warning("Failed to read alpha frame: %s", alpha_path)
        return None

    orig_h, orig_w = img.shape[:2]

    return FramePacket(
        index=index,
        input_stem=input_stem,
        img_raw=img,
        mask_raw=mask,
        orig_h=orig_h,
        orig_w=orig_w,
    )


# ---- Write helpers ----
#
# Writer design: all numpy prep work (color conversion, dtype cast, clipping)
# is done by the CALLER before submitting to the writer pool.  Writer threads
# only call cv2.imwrite on pre-prepared data.  cv2.imwrite is pure C code
# that releases the GIL, so ThreadPool writers achieve real parallelism
# without the IPC serialization cost of ProcessPool.

# EXR write params — module-level to avoid re-creating per call
_EXR_FLAGS = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION,
    cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]


# Sentinel: when color_conversion is _PNG_PREP, the writer does
# RGB float32 → BGR uint8 conversion instead of a simple cvtColor.
_PNG_PREP = -1
# Sentinel: RGBA float32 → BGRA uint8 for transparent PNG output.
_PNG_PREP_RGBA = -2


def _write_frame_outputs(
    write_list: list[tuple[np.ndarray, str, list[int] | None, int | None]],
) -> int:
    """Prepare and write all output files for a single frame.

    Each entry is ``(data, path, imwrite_params, color_conversion)``.
    All heavy work runs here in the writer thread — cv2.cvtColor,
    cv2.convertScaleAbs, and cv2.imwrite are all C code that releases
    the GIL, so writer threads truly parallelize.  The inference thread
    submits raw numpy arrays with zero prep work.

    Returns total bytes written to disk.
    """
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    total_bytes = 0
    for data, path, params, color_conv in write_list:
        if color_conv == _PNG_PREP_RGBA:
            # float32 RGBA → uint8 BGRA (all C calls, GIL-free)
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
            data = cv2.convertScaleAbs(data, alpha=255.0)
        elif color_conv == _PNG_PREP:
            # float32 RGB → uint8 BGR (all C calls, GIL-free)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            data = cv2.convertScaleAbs(data, alpha=255.0)
        elif color_conv is not None:
            data = cv2.cvtColor(data, color_conv)
        if params is not None:
            cv2.imwrite(path, data, params)
        else:
            cv2.imwrite(path, data)
        with contextlib.suppress(OSError):
            total_bytes += os.path.getsize(path)
    return total_bytes


class AsyncInferencePipeline:
    """Multi-stage concurrent pipeline for CorridorKey inference.

    Fully ThreadPool-based — no ProcessPool IPC anywhere.  All heavy work
    (cv2.imread, cv2.cvtColor, cv2.imwrite) is C code that releases the
    GIL, so threads achieve real parallelism.  Inference threads do GPU
    work (GIL released during CUDA kernels) and submit raw numpy arrays
    to writer threads which handle color conversion + encoding.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.engines: list[tuple[str, Any]] = []  # (device_str, engine)
        self._shutdown_event = threading.Event()
        self._warmup_time: float = 0.0  # set by load_engines

    def _warmup_engines(self) -> None:
        """Run a dummy forward pass on each engine to trigger torch.compile.

        GPU 0 compiles first (populates the on-disk FX graph cache with the
        traced graph + compiled Triton binaries).  Remaining GPUs then warmup
        in parallel — if the cache is hit they skip tracing and inductor
        lowering entirely, finishing in seconds instead of minutes.
        """

        def _warmup_one(dev_str: str, engine: Any) -> None:
            if not dev_str.startswith("cuda"):
                return
            torch.cuda.set_device(int(dev_str.split(":")[1]) if ":" in dev_str else 0)
            logger.info("Compiling kernels for %s...", dev_str)
            dummy = np.zeros((self.config.img_size, self.config.img_size, 4), dtype=np.float32)
            t0 = time.perf_counter()
            with contextlib.suppress(Exception):
                engine.process_prepared(dummy, self.config.img_size, self.config.img_size)
            logger.info("Kernel compilation for %s done in %.1fs", dev_str, time.perf_counter() - t0)

            # Capture CUDA graph after torch.compile warmup if requested
            if engine.config.cuda_graphs and engine._cuda_graph is None:
                try:
                    engine.capture_cuda_graph()
                except Exception as e:
                    logger.warning("CUDA graph capture failed on %s: %s", dev_str, e)

        cuda_engines = [(d, e) for d, e in self.engines if d.startswith("cuda")]
        if not cuda_engines:
            return

        # Compile on GPU 0 first.  If the FX cache is warm this is fast
        # (~5s); if cold it does a full compile (~25s) and populates the
        # cache.  Either way, remaining GPUs then warmup in parallel and
        # are guaranteed cache hits — no redundant compilations.
        #
        # On a 20-GPU farm:
        #   Cache warm  → ~5s (GPU 0) + ~5s (GPUs 1-19 parallel) = ~10s
        #   Cache cold  → ~25s (GPU 0 compile) + ~5s (GPUs 1-19 parallel) = ~30s
        _warmup_one(*cuda_engines[0])

        if len(cuda_engines) > 1:
            threads = []
            for dev_str, engine in cuda_engines[1:]:
                t = threading.Thread(target=_warmup_one, args=(dev_str, engine), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

    def load_engines(self) -> None:
        """Discover GPUs, load one engine per device, and warmup torch.compile.

        With fp16 + optimizations, peak VRAM is ~1.5 GB at 2048x2048, so
        virtually any modern GPU works.  Use ``--devices`` to restrict which
        GPUs are used if needed.
        """
        from CorridorKeyModule.backend import create_engine

        devices = self.config.devices

        if devices is None:
            if not torch.cuda.is_available():
                devices = ["cpu"]
            else:
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    free, total = torch.cuda.mem_get_info(i)
                    logger.info(
                        "GPU %d (%s): %.0f MB free / %.0f MB total",
                        i,
                        name,
                        free / (1024 * 1024),
                        total / (1024 * 1024),
                    )
                    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

                if not devices:
                    logger.info("No CUDA GPUs found. Falling back to CPU.")
                    devices = ["cpu"]

        logger.info("Loading engines on devices: %s", devices)
        t_load0 = time.perf_counter()
        for dev in devices:
            engine = create_engine(
                backend=self.config.backend,
                device=dev,
                img_size=self.config.img_size,
                optimization_config=self.config.optimization_config,
            )
            self.engines.append((dev, engine))
            logger.info("Engine loaded on %s", dev)

        self._warmup_engines()
        self._warmup_time = time.perf_counter() - t_load0

    def process_clip(
        self,
        input_paths: list[str],
        alpha_paths: list[str],
        input_stems: list[str],
        output_dirs: dict[str, str],
        settings: InferenceSettings,
        max_frames: int | None = None,
        on_progress: Callable[[int, int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run the async pipeline on a single clip.

        Parameters
        ----------
        input_paths : list[str]
            Absolute paths to input frames (image files).
        alpha_paths : list[str]
            Absolute paths to alpha hint frames (image files).
        input_stems : list[str]
            Filename stems for output files.
        output_dirs : dict
            Keys: "fg", "matte", "comp", "processed" → directory paths.
        settings : InferenceSettings
            User-configured inference parameters.
        max_frames : int | None
            Limit frames processed (for testing).

        Returns
        -------
        dict with "total", "completed", "failed", "skipped_frames".
        """
        num_frames = min(len(input_paths), len(alpha_paths))
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        if num_frames == 0:
            return {"total": 0, "completed": 0, "failed": 0, "skipped_frames": []}

        num_gpus = len(self.engines)
        prefetch = self.config.prefetch_depth or (num_gpus * 8)

        # Auto-scale workers based on GPU count and available cores.
        #
        # Both pools are ThreadPool — cv2.imread and cv2.imwrite are C code
        # that releases the GIL, so threads achieve real parallelism.
        #
        # Readers need very few threads: cv2.imread is I/O-bound and the
        # work_q backpressure naturally throttles them.  Over-provisioning
        # readers wastes disk bandwidth that writers need.
        #
        # Writers are the bottleneck: EXR PXR24 compression is CPU-heavy.
        # Give them everything else.  Each GPU gets its own writer pool
        # (created below), so num_writers here is the per-GPU count.
        cores = os.cpu_count() or 4

        if self.config.read_workers:
            num_readers = self.config.read_workers
        else:
            # Readers: one per GPU is enough — inputs are small compressed
            # files and cv2.imread is GIL-free.  Backpressure from work_q
            # throttles them naturally.  Minimum 2 for pipelining.
            num_readers = max(2, num_gpus)

        if self.config.write_workers:
            num_writers = self.config.write_workers
        else:
            # Writers get all remaining cores — EXR PXR24 compression is
            # CPU-heavy and dominates IO time.  Floor: at least as many
            # writers as readers so writes never starve.
            num_writers = max(num_readers, cores - num_readers)

        logger.info(
            "Worker pool: %d readers, %d writers (%d cores, %d GPUs)",
            num_readers,
            num_writers,
            os.cpu_count() or 0,
            num_gpus,
        )

        # Tracking
        completed_count = [0]
        failed_frames: list[int] = []
        failed_lock = threading.Lock()
        pipeline_t0 = time.perf_counter()
        profiler = _TimelineProfiler(pipeline_t0)

        # IO byte counters (atomic via lock-free int adds on CPython)
        io_bytes_read = [0]
        io_bytes_written = [0]

        self._shutdown_event.clear()

        # --- Progress callback (non-blocking) ---
        # The callback is invoked from the progress/drain thread; Rich
        # rendering must never stall write completion, so we fire it on
        # a separate coalescing thread that merges rapid updates.
        _progress_pending = threading.Event()
        _progress_done = threading.Event()

        def _report_progress() -> None:
            if on_progress is not None:
                _progress_pending.set()

        def _progress_callback_worker() -> None:
            """Coalesce progress updates so Rich never blocks the drain."""
            while not _progress_done.is_set():
                if _progress_pending.wait(timeout=0.1):
                    _progress_pending.clear()
                    with contextlib.suppress(Exception):
                        on_progress(completed_count[0], num_frames,
                                    io_bytes_read[0], io_bytes_written[0])
            # Final update
            if on_progress is not None:
                with contextlib.suppress(Exception):
                    on_progress(completed_count[0], num_frames,
                                io_bytes_read[0], io_bytes_written[0])

        _cb_thread: threading.Thread | None = None
        if on_progress is not None:
            _cb_thread = threading.Thread(
                target=_progress_callback_worker, name="progress-callback", daemon=True,
            )
            _cb_thread.start()

        # --- Flow control ---
        #
        # Backpressure chain: disk ← write_pool ← write_q ← inference ← work_q ← readers
        #
        # work_q (maxsize=prefetch) throttles readers: when GPUs can't keep up,
        # the queue fills and readers block on put(), naturally pacing disk reads.
        #
        # Write concurrency is bounded by write_pool's max_workers — no
        # semaphore needed.  The write drain thread submits freely; excess
        # tasks queue inside the ThreadPoolExecutor which handles its own
        # backpressure.  This avoids any blocking on the drain thread.

        # Memory headroom: stop reading when available RAM would drop
        # below this threshold.  Resumes when drain frees enough memory.
        _MEM_HEADROOM = 1 * 1024 * 1024 * 1024  # 1 GB
        _last_frame_bytes = [0]  # estimated from last completed read

        def _available_ram() -> int:
            """Available system memory in bytes (Linux fast path)."""
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except OSError:
                pass
            # Fallback
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

        def reader_task(read_pool: ThreadPoolExecutor) -> None:
            """Submit frame reads incrementally, feed work_q as reads complete.

            Submits at most `prefetch` reads ahead of consumption, and
            pauses when available system RAM minus one frame's estimated
            size would drop below 1 GB.  Resumes when the drain frees
            memory by writing and releasing resolved frames.
            """
            next_submit = 0
            in_flight: dict[Any, int] = {}  # future → frame index

            def _mem_ok() -> bool:
                """True if we have enough RAM to decode another frame."""
                if _last_frame_bytes[0] == 0:
                    return True  # no estimate yet, allow first reads
                return _available_ram() - _last_frame_bytes[0] > _MEM_HEADROOM

            def _submit_batch() -> None:
                """Submit reads up to prefetch limit, respecting RAM."""
                nonlocal next_submit
                while next_submit < num_frames and len(in_flight) < prefetch:
                    if self._shutdown_event.is_set():
                        break
                    if not _mem_ok():
                        break
                    f = read_pool.submit(
                        _read_frame_pair,
                        input_paths[next_submit],
                        alpha_paths[next_submit],
                        next_submit,
                        input_stems[next_submit],
                    )
                    in_flight[f] = next_submit
                    next_submit += 1

            _submit_batch()

            while in_flight:
                if self._shutdown_event.is_set():
                    break
                # Wait for any one read to complete
                done = next(as_completed(in_flight))
                i = in_flight.pop(done)
                try:
                    packet = done.result()
                    if packet is not None:
                        # Update frame size estimate
                        _last_frame_bytes[0] = packet.img_raw.nbytes + packet.mask_raw.nbytes
                        # Track IO bytes read (file sizes on disk)
                        with contextlib.suppress(OSError):
                            io_bytes_read[0] += (
                                os.path.getsize(input_paths[i])
                                + os.path.getsize(alpha_paths[i])
                            )
                        profiler.mark("reader", "read_done", i)
                        work_q.put(packet)
                    else:
                        with failed_lock:
                            failed_frames.append(i)
                except Exception as e:
                    logger.warning("Read error frame %d: %s", i, e)
                    with failed_lock:
                        failed_frames.append(i)

                # Refill: submit more reads now that one slot freed up
                # If memory is tight, wait for drain to free some
                if next_submit < num_frames and not _mem_ok():
                    profiler.mark("reader", "mem_wait", next_submit)
                    while not _mem_ok() and not self._shutdown_event.is_set():
                        threading.Event().wait(0.1)
                _submit_batch()

            # Signal inference threads to stop
            for _ in range(num_gpus):
                work_q.put(_SHUTDOWN)

        inference_done_event = threading.Event()
        write_futures: list[tuple[int, Any]] = []
        write_futures_lock = threading.Lock()

        # Write queue decouples inference threads from post-inference work.
        # The inference thread only does GPU work and passes
        # (packet, PendingTransfer) to write workers via write_q.
        write_q: queue.Queue[tuple[FramePacket, Any] | None] = queue.Queue()

        def _resolve_and_submit(item: tuple[FramePacket, Any], write_pool: ThreadPoolExecutor) -> None:
            """Resolve one DMA transfer and submit file writes to the pool."""
            packet, transfer_or_result = item
            fidx = packet.index

            # Resolve PendingTransfer → numpy dict
            t_res = profiler.span_begin("write_drain", "dma_resolve", fidx)
            if hasattr(transfer_or_result, "resolve"):
                result = transfer_or_result.resolve()
            else:
                result = transfer_or_result
            profiler.span_end("write_drain", "dma_resolve", t_res, fidx)

            stem = packet.input_stem
            alpha = result["alpha"]
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]

            write_list: list[tuple[np.ndarray, str, list[int] | None, int | None]] = [
                (result["fg"], os.path.join(output_dirs["fg"], f"{stem}.exr"), _EXR_FLAGS, cv2.COLOR_RGB2BGR),
                (alpha, os.path.join(output_dirs["matte"], f"{stem}.exr"), _EXR_FLAGS, None),
            ]
            comp = result.get("comp")
            if self.config.output_comp_png and comp is not None:
                png_prep = _PNG_PREP_RGBA if comp.ndim == 3 and comp.shape[2] == 4 else _PNG_PREP
                write_list.append(
                    (comp, os.path.join(output_dirs["comp"], f"{stem}.png"), None, png_prep)
                )
            processed = result.get("processed")
            if processed is not None:
                write_list.append(
                    (processed, os.path.join(output_dirs["processed"], f"{stem}.exr"), _EXR_FLAGS, cv2.COLOR_RGBA2BGRA)
                )

            fut = write_pool.submit(_write_frame_outputs, write_list)
            with write_futures_lock:
                write_futures.append((fidx, fut))

        def _drain_worker(write_pool: ThreadPoolExecutor) -> None:
            """Persistent drain worker — pulls items from write_q until None.

            Multiple drain workers run concurrently (num_gpus of them),
            giving one resolve in flight per GPU.  Each does DMA sync +
            numpy copies, then hands file writes to the write pool.
            """
            while True:
                item = write_q.get()
                if item is None:
                    # Put sentinel back for the next worker to see
                    write_q.put(None)
                    break
                _resolve_and_submit(item, write_pool)

        def inference_worker(device_str: str, engine: Any, write_pool: ThreadPoolExecutor) -> None:
            """Pull frames from work_q, run deferred-DMA inference, hand off to writer.

            Uses process_raw_deferred so the GPU→CPU DMA runs on the copy
            stream while the next frame's forward pass runs on the compute
            stream.  The PendingTransfer is passed directly to the write
            drain thread — the inference thread never calls resolve() and
            never blocks on numpy copies.
            """
            gpu_label = f"gpu:{device_str}"

            if device_str.startswith("cuda"):
                dev_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                torch.cuda.set_device(dev_idx)

            first_frame = True
            pending_transfer = None  # PendingTransfer from previous frame
            pending_packet = None  # FramePacket from previous frame

            while not self._shutdown_event.is_set():
                t_wait = profiler.span_begin(gpu_label, "queue_wait")
                item = work_q.get()
                profiler.span_end(gpu_label, "queue_wait", t_wait)
                if item is _SHUTDOWN:
                    break

                packet: FramePacket = item
                fidx = packet.index
                if first_frame:
                    logger.info(
                        "%s: first frame received (idx=%d, %dx%d)", device_str, fidx, packet.orig_w, packet.orig_h
                    )

                try:
                    t_fwd = profiler.span_begin(gpu_label, "forward", fidx)
                    transfer = engine.process_raw_deferred(
                        packet.img_raw,
                        packet.mask_raw,
                        packet.orig_h,
                        packet.orig_w,
                        input_is_linear=settings.input_is_linear,
                        fg_is_straight=True,
                        despill_strength=settings.despill_strength,
                        auto_despeckle=settings.auto_despeckle,
                        despeckle_size=settings.despeckle_size,
                        refiner_scale=settings.refiner_scale,
                    )
                    profiler.span_end(gpu_label, "forward", t_fwd, fidx)

                    if first_frame:
                        logger.info("%s: first frame inference done", device_str)
                        first_frame = False

                    # Hand off PREVIOUS frame's PendingTransfer to the write
                    # drain thread.  DMA resolve + numpy copies happen there,
                    # keeping this thread free for GPU work.
                    if pending_transfer is not None:
                        write_q.put((pending_packet, pending_transfer))

                    # Rotate: current becomes pending
                    pending_transfer = transfer
                    pending_packet = packet

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "\033[1mCUDA OOM on %s for frame %d — taking GPU offline\033[0m",
                        device_str,
                        packet.index,
                    )
                    torch.cuda.empty_cache()
                    # Drain pending before going offline
                    if pending_transfer is not None:
                        write_q.put((pending_packet, pending_transfer))
                        pending_transfer = None
                    with failed_lock:
                        failed_frames.append(packet.index)
                    break
                except Exception as e:
                    logger.warning("Inference error frame %d on %s: %s", packet.index, device_str, e)
                    with failed_lock:
                        failed_frames.append(packet.index)

            # Drain: hand off the last frame's PendingTransfer
            if pending_transfer is not None:
                write_q.put((pending_packet, pending_transfer))

        # --- Writer/progress stage (resolves write futures, updates progress) ---
        def progress_task() -> None:
            """Wait for write futures to complete and update progress bar."""
            resolved = 0
            target = num_frames

            while resolved < target:
                if self._shutdown_event.is_set():
                    break

                with write_futures_lock:
                    pending = list(write_futures)

                newly_done = []
                for frame_idx, fut in pending:
                    if fut.done():
                        newly_done.append((frame_idx, fut))

                if newly_done:
                    with write_futures_lock:
                        for item in newly_done:
                            write_futures.remove(item)

                    for frame_idx, fut in newly_done:
                        try:
                            written = fut.result()
                            if isinstance(written, int):
                                io_bytes_written[0] += written
                            completed_count[0] += 1
                            profiler.frame_completed()
                        except Exception as e:
                            logger.warning("Write error frame %d: %s", frame_idx, e)
                            with failed_lock:
                                failed_frames.append(frame_idx)
                        resolved += 1
                        _report_progress()
                else:
                    if inference_done_event.is_set():
                        with write_futures_lock:
                            if not write_futures:
                                break
                    threading.Event().wait(0.01)

        # --- Launch pipeline ---
        read_pool = ThreadPoolExecutor(max_workers=num_readers)
        write_pool = ThreadPoolExecutor(max_workers=num_writers)

        work_q: queue.Queue[FramePacket | object] = queue.Queue(maxsize=prefetch)

        reader_thread = threading.Thread(target=reader_task, args=(read_pool,), name="pipeline-reader", daemon=True)
        progress_thread = threading.Thread(target=progress_task, name="pipeline-progress", daemon=True)

        # N persistent drain workers (one per GPU) pull from write_q.
        # Bounded concurrency: at most num_gpus resolves in flight.
        drain_threads = [
            threading.Thread(
                target=_drain_worker,
                args=(write_pool,),
                name=f"pipeline-drain-{i}",
                daemon=True,
            )
            for i in range(num_gpus)
        ]

        inference_threads = []
        for device_str, engine in self.engines:
            t = threading.Thread(
                target=inference_worker,
                args=(device_str, engine, write_pool),
                name=f"pipeline-infer-{device_str}",
                daemon=True,
            )
            inference_threads.append(t)

        reader_done_t = inference_done_t = pipeline_done_t = pipeline_t0
        completed_at_infer_done = 0

        try:
            reader_thread.start()
            for dt in drain_threads:
                dt.start()
            for t in inference_threads:
                t.start()
            progress_thread.start()

            # Wait for reader to finish submitting
            reader_thread.join()
            reader_done_t = time.perf_counter()

            # Wait for all inference threads
            for t in inference_threads:
                t.join()
            inference_done_t = time.perf_counter()
            completed_at_infer_done = completed_count[0]

            # Signal drain workers to stop (one None cascades via put-back)
            write_q.put(None)
            for dt in drain_threads:
                dt.join()

            # Signal progress thread that inference is done
            inference_done_event.set()

            # Wait for all writes to flush
            progress_thread.join()
            pipeline_done_t = time.perf_counter()

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted — shutting down...")
            self._shutdown_event.set()
            inference_done_event.set()

            # Cancel any pending futures
            with write_futures_lock:
                for _, fut in write_futures:
                    fut.cancel()
                write_futures.clear()

            # Drain work queue to unblock inference threads
            while not work_q.empty():
                try:
                    work_q.get_nowait()
                except queue.Empty:
                    break
            for _ in range(num_gpus):
                with contextlib.suppress(queue.Full):
                    work_q.put_nowait(_SHUTDOWN)

            # Stop drain workers
            write_q.put(None)

            reader_thread.join(timeout=10)
            for dt in drain_threads:
                dt.join(timeout=5)
            for t in inference_threads:
                t.join(timeout=10)
            progress_thread.join(timeout=5)

        finally:
            _progress_done.set()
            _progress_pending.set()  # Wake the thread so it can exit
            if _cb_thread is not None:
                _cb_thread.join(timeout=2)
            shutting_down = self._shutdown_event.is_set()
            read_pool.shutdown(wait=not shutting_down, cancel_futures=shutting_down)
            write_pool.shutdown(wait=not shutting_down, cancel_futures=shutting_down)

        # --- Summary ---
        num_failed = len(failed_frames)
        if num_failed > 0:
            failed_frames.sort()
            logger.warning(
                "\033[1m%d frame(s) failed: %s\033[0m",
                num_failed,
                failed_frames[:20] if num_failed > 20 else failed_frames,
            )

        # Report timing
        total_wall = pipeline_done_t - pipeline_t0
        read_wall = reader_done_t - pipeline_t0
        infer_wall = inference_done_t - pipeline_t0
        write_drain = pipeline_done_t - inference_done_t

        # Console summary
        logger.info(
            profiler.console_summary(
                num_frames,
                total_wall,
                self._warmup_time,
                read_wall,
                infer_wall,
                write_drain,
                completed_at_infer_done,
            )
        )

        # Peak VRAM per GPU
        for dev_str, _ in self.engines:
            if dev_str.startswith("cuda"):
                dev_idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
                peak_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024**2)
                logger.info("  Peak VRAM %s: %.0f MB (%.2f GB)", dev_str, peak_mb, peak_mb / 1024)

        # --- Postprocess mode hint ---
        hint = None
        if not self._shutdown_event.is_set() and num_frames > 10:
            opt = self.config.optimization_config
            gpu_pp = getattr(opt, "gpu_postprocess", True) if opt else True
            pct_done = completed_at_infer_done / num_frames
            early_threshold = min(0.75, 125 / num_frames)
            late_threshold = max(0.95, 1.0 - num_writers / num_frames)

            if not gpu_pp and pct_done < early_threshold:
                hint = (
                    f"GPU finished with only {pct_done:.0%} of frames written. "
                    "Consider --gpu-postprocess to offload work from CPU."
                )
            elif gpu_pp and pct_done > late_threshold:
                hint = (
                    f"GPU finished with {pct_done:.0%} of frames already written. "
                    "Consider --cpu-postprocess to free GPU time for inference."
                )

            if hint:
                logger.info("  Hint: %s", hint)

        return {
            "total": num_frames,
            "completed": completed_count[0],
            "failed": num_failed,
            "skipped_frames": sorted(failed_frames),
            "hint": hint,
        }

    def shutdown(self) -> None:
        """Signal all threads to stop."""
        self._shutdown_event.set()
