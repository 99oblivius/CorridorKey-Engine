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

import dataclasses
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cv2
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Sentinel to signal shutdown
_SHUTDOWN = object()



class _StageProfiler:
    """Lightweight per-GPU stage profiler.

    Tracks cumulative time and idle time for each stage of the inference
    worker (queue wait, tensor upload, inference, postprocess, write submit).
    Printed as a summary at the end of the run.
    """

    def __init__(self, device_str: str):
        self.device = device_str
        self.counts: dict[str, int] = {}
        self.totals: dict[str, float] = {}
        self._t0: float = 0.0

    def begin(self, stage: str) -> None:
        self._t0 = time.perf_counter()

    def end(self, stage: str) -> None:
        elapsed = time.perf_counter() - self._t0
        self.totals[stage] = self.totals.get(stage, 0.0) + elapsed
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def summary(self) -> str:
        lines = [f"  Pipeline profile for {self.device}:"]
        total = sum(self.totals.values())
        for stage, t in self.totals.items():
            n = self.counts.get(stage, 0)
            avg_ms = (t / n * 1000) if n > 0 else 0
            pct = (t / total * 100) if total > 0 else 0
            lines.append(f"    {stage:20s}: {t:7.2f}s total | {avg_ms:7.1f}ms avg | {pct:5.1f}%  ({n} calls)")
        lines.append(f"    {'TOTAL':20s}: {total:7.2f}s")
        return "\n".join(lines)



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


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"))



def _read_input_frame(
    path: str, is_exr: bool,
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
    else:
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
    elif mask.dtype == np.uint16:
        return mask.astype(np.float32) / 65535.0
    else:
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


def _read_video_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read one frame from a VideoCapture, return RGB float32 or None."""
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


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


def _write_frame_outputs(
    write_list: list[tuple[np.ndarray, str, list[int] | None, int | None]],
) -> None:
    """Prepare and write all output files for a single frame.

    Each entry is ``(data, path, imwrite_params, color_conversion)``.
    All heavy work runs here in the writer thread — cv2.cvtColor,
    cv2.convertScaleAbs, and cv2.imwrite are all C code that releases
    the GIL, so writer threads truly parallelize.  The inference thread
    submits raw numpy arrays with zero prep work.
    """
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    for data, path, params, color_conv in write_list:
        if color_conv == _PNG_PREP:
            # float32 RGB → uint8 BGR (all C calls, GIL-free)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            data = cv2.convertScaleAbs(data, alpha=255.0)
        elif color_conv is not None:
            data = cv2.cvtColor(data, color_conv)
        if params is not None:
            cv2.imwrite(path, data, params)
        else:
            cv2.imwrite(path, data)


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
            try:
                engine.process_prepared(dummy, self.config.img_size, self.config.img_size)
            except Exception:
                pass
            logger.info("Kernel compilation for %s done in %.1fs", dev_str, time.perf_counter() - t0)

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
                        i, name, free / (1024 * 1024), total / (1024 * 1024),
                    )
                    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

                if not devices:
                    logger.info("No CUDA GPUs found. Falling back to CPU.")
                    devices = ["cpu"]

        logger.info("Loading engines on devices: %s", devices)
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

    def process_clip(
        self,
        input_paths: list[str],
        alpha_paths: list[str],
        input_stems: list[str],
        output_dirs: dict[str, str],
        settings: InferenceSettings,
        max_frames: int | None = None,
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
            # Generous readers: cv2.imread is GIL-free C code, so more
            # threads = more parallel decoding with no GIL contention.
            # work_q backpressure prevents memory buildup.  Pre-emptive
            # reading keeps the queue full so GPUs never stall between
            # batches.
            num_readers = max(4, num_gpus * 4)

        if self.config.write_workers:
            num_writers = self.config.write_workers
        else:
            # Writers get all remaining cores.  Per-GPU pools split this
            # total across GPUs (done below).
            num_writers = max(4, cores - num_readers)

        logger.info("Worker pool: %d readers, %d writers (%d cores, %d GPUs)",
                     num_readers, num_writers, os.cpu_count() or 0, num_gpus)

        # Tracking
        completed_count = [0]
        failed_frames: list[int] = []
        failed_lock = threading.Lock()

        self._shutdown_event.clear()

        # --- Progress bar ---
        from tqdm import tqdm

        pbar = tqdm(total=num_frames, desc="CorridorKey", unit="frame", dynamic_ncols=True)

        # --- Flow control ---
        #
        # Backpressure chain: disk ← writers ← write_sem ← inference ← work_q ← readers
        #
        # work_q (maxsize=prefetch) throttles readers: when GPUs can't keep up,
        # the queue fills and readers block on put(), naturally pacing disk reads.
        #
        # Write backpressure: a shared semaphore limits in-flight writes.
        # When writers can't flush to disk fast enough, inference threads
        # block before submitting new writes, which backs up into work_q
        # and naturally paces reads.
        #
        # A shared pool (not per-GPU) avoids starving faster GPUs when
        # GPU speeds are mismatched — the fast GPU can use idle writers
        # from the slow GPU's share.
        write_sem = threading.Semaphore(num_writers * 2)

        def reader_task(read_pool: ThreadPoolExecutor):
            """Submit frame reads to thread pool, feed work_q as reads complete.

            All frames are submitted to the pool upfront.  Backpressure comes
            from work_q (maxsize=prefetch): when the queue fills, the
            as_completed loop blocks on put(), which is fine — reader threads
            in the pool keep decoding ahead but only num_readers are active
            at any time, limiting disk I/O contention with writers.
            """
            future_to_index = {}
            for i in range(num_frames):
                if self._shutdown_event.is_set():
                    break
                f = read_pool.submit(
                    _read_frame_pair,
                    input_paths[i],
                    alpha_paths[i],
                    i,
                    input_stems[i],
                )
                future_to_index[f] = i

            # Feed work_q as reads complete (not in order — avoids head-of-line blocking)
            for f in as_completed(future_to_index):
                if self._shutdown_event.is_set():
                    break
                i = future_to_index[f]
                try:
                    packet = f.result()
                    if packet is not None:
                        work_q.put(packet)
                    else:
                        with failed_lock:
                            failed_frames.append(i)
                except Exception as e:
                    logger.warning("Read error frame %d: %s", i, e)
                    with failed_lock:
                        failed_frames.append(i)

            # Signal inference threads to stop
            for _ in range(num_gpus):
                work_q.put(_SHUTDOWN)

        inference_done_event = threading.Event()
        write_futures: list[tuple[int, Any]] = []
        write_futures_lock = threading.Lock()

        profilers: list[_StageProfiler] = []
        profilers_lock = threading.Lock()

        def _submit_write(packet: FramePacket, result: dict, write_pool: ThreadPoolExecutor, prof: _StageProfiler) -> None:
            """Build write list from result dict and submit to writer pool."""
            prof.begin("write_submit")
            stem = packet.input_stem
            alpha = result["alpha"]
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]

            write_list: list[tuple[np.ndarray, str, list[int] | None, int | None]] = [
                (result["fg"], os.path.join(output_dirs["fg"], f"{stem}.exr"),
                 _EXR_FLAGS, cv2.COLOR_RGB2BGR),
                (alpha, os.path.join(output_dirs["matte"], f"{stem}.exr"),
                 _EXR_FLAGS, None),
                (result["comp"], os.path.join(output_dirs["comp"], f"{stem}.png"),
                 None, _PNG_PREP),
            ]
            processed = result.get("processed")
            if processed is not None:
                write_list.append(
                    (processed, os.path.join(output_dirs["processed"], f"{stem}.exr"),
                     _EXR_FLAGS, cv2.COLOR_RGBA2BGRA)
                )

            write_sem.acquire()
            fut = write_pool.submit(_write_frame_outputs, write_list)
            with write_futures_lock:
                write_futures.append((packet.index, fut))
            prof.end("write_submit")

        def inference_worker(device_str: str, engine: Any, write_pool: ThreadPoolExecutor):
            """Pull frames from work_q, run deferred-DMA inference, submit writes.

            Uses process_raw_deferred so the GPU→CPU DMA transfer for
            frame N overlaps with frame N+1's preprocess + forward pass.
            The previous frame's transfer is resolved (synced) just before
            submitting its write job.
            """
            prof = _StageProfiler(device_str)
            with profilers_lock:
                profilers.append(prof)

            if device_str.startswith("cuda"):
                dev_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                torch.cuda.set_device(dev_idx)

            first_frame = True
            pending_transfer = None  # PendingTransfer from previous frame
            pending_packet = None    # FramePacket from previous frame

            while not self._shutdown_event.is_set():
                prof.begin("queue_wait")
                item = work_q.get()
                prof.end("queue_wait")
                if item is _SHUTDOWN:
                    break

                packet: FramePacket = item
                if first_frame:
                    logger.info("%s: first frame received (idx=%d, %dx%d)",
                                device_str, packet.index, packet.orig_w, packet.orig_h)

                try:
                    timings: dict = {}
                    t_inf0 = time.perf_counter()
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
                        _timings=timings,
                    )
                    if first_frame:
                        logger.info("%s: first frame done in %.1fs (preprocess=%.3f forward=%.3f postprocess=%.3f)",
                                    device_str, time.perf_counter() - t_inf0,
                                    timings.get("preprocess", 0), timings.get("forward", 0), timings.get("postprocess", 0))
                        first_frame = False

                    # Record sub-stage timings
                    for stage, elapsed in timings.items():
                        prof.begin(stage)
                        prof._t0 = time.perf_counter() - elapsed
                        prof.end(stage)

                    # Resolve PREVIOUS frame's DMA (overlapped with current
                    # frame's preprocess + forward on the compute stream).
                    if pending_transfer is not None:
                        prof.begin("dma_resolve")
                        result = pending_transfer.resolve()
                        prof.end("dma_resolve")
                        _submit_write(pending_packet, result, write_pool, prof)

                    # Rotate: current becomes pending
                    pending_transfer = transfer
                    pending_packet = packet

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "\033[1mCUDA OOM on %s for frame %d — taking GPU offline\033[0m",
                        device_str, packet.index,
                    )
                    torch.cuda.empty_cache()
                    # Drain pending before going offline
                    if pending_transfer is not None:
                        try:
                            result = pending_transfer.resolve()
                            _submit_write(pending_packet, result, write_pool, prof)
                        except Exception:
                            with failed_lock:
                                failed_frames.append(pending_packet.index)
                        pending_transfer = None
                    with failed_lock:
                        failed_frames.append(packet.index)
                    break
                except Exception as e:
                    logger.warning("Inference error frame %d on %s: %s", packet.index, device_str, e)
                    with failed_lock:
                        failed_frames.append(packet.index)

            # Drain: resolve the last frame's DMA
            if pending_transfer is not None:
                try:
                    prof.begin("dma_resolve")
                    result = pending_transfer.resolve()
                    prof.end("dma_resolve")
                    _submit_write(pending_packet, result, write_pool, prof)
                except Exception as e:
                    logger.warning("Final DMA resolve error frame %d on %s: %s",
                                   pending_packet.index, device_str, e)
                    with failed_lock:
                        failed_frames.append(pending_packet.index)

        # --- Writer/progress stage (resolves write futures, updates progress) ---
        def progress_task():
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
                            fut.result()
                            completed_count[0] += 1
                        except Exception as e:
                            logger.warning("Write error frame %d: %s", frame_idx, e)
                            with failed_lock:
                                failed_frames.append(frame_idx)
                        finally:
                            write_sem.release()
                        resolved += 1
                        pbar.update(1)
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

        inference_threads = []
        for device_str, engine in self.engines:
            t = threading.Thread(
                target=inference_worker,
                args=(device_str, engine, write_pool),
                name=f"pipeline-infer-{device_str}",
                daemon=True,
            )
            inference_threads.append(t)

        pipeline_t0 = time.perf_counter()
        reader_done_t = inference_done_t = pipeline_done_t = pipeline_t0

        try:
            reader_thread.start()
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
                try:
                    work_q.put_nowait(_SHUTDOWN)
                except queue.Full:
                    pass

            reader_thread.join(timeout=10)
            for t in inference_threads:
                t.join(timeout=10)
            progress_thread.join(timeout=5)

        finally:
            pbar.close()
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

        # Report pipeline-level timing
        total_wall = pipeline_done_t - pipeline_t0
        read_wall = reader_done_t - pipeline_t0
        infer_wall = inference_done_t - pipeline_t0
        write_drain = pipeline_done_t - inference_done_t
        logger.info(
            "\n  Pipeline wall-clock breakdown:\n"
            "    read phase       : %7.2fs  (readers submitting to queue)\n"
            "    inference phase  : %7.2fs  (all GPUs done)\n"
            "    write drain      : %7.2fs  (flushing remaining EXR/PNG writes)\n"
            "    total            : %7.2fs  (%.1f fps)",
            read_wall, infer_wall, write_drain, total_wall,
            num_frames / total_wall if total_wall > 0 else 0,
        )

        # Report per-GPU profiling
        for prof in profilers:
            logger.info("\n%s", prof.summary())

        # Report peak VRAM per GPU
        for dev_str, _ in self.engines:
            if dev_str.startswith("cuda"):
                dev_idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
                peak_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024**2)
                logger.info("Peak VRAM %s: %.0f MB (%.2f GB)", dev_str, peak_mb, peak_mb / 1024)

        return {
            "total": num_frames,
            "completed": completed_count[0],
            "failed": num_failed,
            "skipped_frames": sorted(failed_frames),
        }

    def shutdown(self) -> None:
        """Signal all threads to stop."""
        self._shutdown_event.set()
