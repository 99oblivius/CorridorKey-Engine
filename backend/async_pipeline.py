"""Async pipelined inference engine for CorridorKey.

Overlaps frame I/O with GPU inference using a multi-stage concurrent
pipeline. Supports multiple GPUs via work-stealing.

Architecture:
    Read Workers (ThreadPool) → Work Queue → Inference Threads (1:1 per GPU)
    → Write Queue → Writer Thread (single, serialized for EXR safety)

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Sentinel to signal shutdown
_SHUTDOWN = object()


@dataclasses.dataclass
class PipelineConfig:
    """Tunable pipeline parameters."""

    img_size: int = 2048
    prefetch_depth: int = 0  # 0 = auto (num_gpus * 4)
    read_workers: int = 0  # 0 = auto (cpu_count // 2, min 2)
    write_workers: int = 1  # Always 1 (EXR not thread-safe)
    devices: list[str] | None = None  # None = auto-detect
    backend: str | None = None  # "torch", "mlx", or None for auto


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
    """Work item flowing through the pipeline."""

    index: int  # Original frame order
    input_stem: str  # Output filename stem
    img_srgb: np.ndarray  # Input image (H,W,3) float32 0-1
    mask_linear: np.ndarray  # Alpha hint (H,W) float32 0-1


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


def enumerate_usable_gpus(min_free_vram_mb: int) -> list[str]:
    """Return list of CUDA device strings with enough free VRAM."""
    if not torch.cuda.is_available():
        return []

    usable = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_mb = free / (1024 * 1024)
        name = torch.cuda.get_device_name(i)
        logger.info("GPU %d (%s): %.0f MB free / %.0f MB total", i, name, free_mb, total / (1024 * 1024))
        if free_mb >= min_free_vram_mb:
            usable.append(f"cuda:{i}")
        else:
            logger.warning("GPU %d (%s): insufficient VRAM (%.0f MB free, need %d MB)", i, name, free_mb, min_free_vram_mb)
    return usable


def estimate_vram_mb(img_size: int) -> int:
    """Rough VRAM estimate based on resolution.

    The model weights are ~300MB, but activations at 2048x2048 with
    autocast FP16 peak at ~22GB. We use a tight margin because the
    engine uses torch.autocast(float16) which halves activation memory.
    """
    # Empirical: 2048 needs ~22000 MB with autocast, scales ~quadratically
    base_size = 2048
    base_vram = 22000
    scale = (img_size / base_size) ** 2
    return int(base_vram * scale)


def _read_input_frame(
    path: str, is_exr: bool, input_is_linear: bool
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
    input_is_linear: bool,
) -> FramePacket | None:
    """Read one input+alpha pair and return a FramePacket. Returns None on failure."""
    is_exr = input_path.lower().endswith(".exr")
    img = _read_input_frame(input_path, is_exr, input_is_linear)
    if img is None:
        logger.warning("Failed to read input frame: %s", input_path)
        return None

    mask = _read_alpha_frame(alpha_path)
    if mask is None:
        logger.warning("Failed to read alpha frame: %s", alpha_path)
        return None

    # Resize mask to match input if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    return FramePacket(index=index, input_stem=input_stem, img_srgb=img, mask_linear=mask)


def _read_video_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read one frame from a VideoCapture, return RGB float32 or None."""
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _write_result(
    result: ResultPacket,
    fg_dir: str,
    matte_dir: str,
    comp_dir: str,
    proc_dir: str,
) -> None:
    """Write all outputs for one frame. Must be called from a single thread."""
    exr_flags = [
        cv2.IMWRITE_EXR_TYPE,
        cv2.IMWRITE_EXR_TYPE_HALF,
        cv2.IMWRITE_EXR_COMPRESSION,
        cv2.IMWRITE_EXR_COMPRESSION_PXR24,
    ]

    stem = result.input_stem

    # FG (sRGB float → BGR EXR)
    fg_bgr = cv2.cvtColor(result.fg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(fg_dir, f"{stem}.exr"), fg_bgr, exr_flags)

    # Matte (single channel linear float EXR)
    alpha = result.alpha
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    cv2.imwrite(os.path.join(matte_dir, f"{stem}.exr"), alpha, exr_flags)

    # Comp (PNG 8-bit)
    comp_bgr = cv2.cvtColor(
        (np.clip(result.comp, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(os.path.join(comp_dir, f"{stem}.png"), comp_bgr)

    # Processed (RGBA EXR)
    if result.processed is not None:
        proc_bgra = cv2.cvtColor(result.processed, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(proc_dir, f"{stem}.exr"), proc_bgra, exr_flags)


class AsyncInferencePipeline:
    """Multi-stage concurrent pipeline for CorridorKey inference."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.engines: list[tuple[str, Any]] = []  # (device_str, engine)
        self._shutdown_event = threading.Event()

    def load_engines(self) -> None:
        """Discover GPUs and load one engine per usable device.

        Checks free VRAM before each engine load to account for VRAM
        consumed by previously loaded engines and other processes.
        """
        from CorridorKeyModule.backend import create_engine

        devices = self.config.devices
        min_vram = estimate_vram_mb(self.config.img_size)

        if devices is None:
            if not torch.cuda.is_available():
                devices = ["cpu"]
            else:
                # Load engines one at a time, re-checking VRAM before each
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    free_mb = free / (1024 * 1024)
                    name = torch.cuda.get_device_name(i)
                    logger.info(
                        "GPU %d (%s): %.0f MB free / %.0f MB total (need %d MB)",
                        i, name, free_mb, total / (1024 * 1024), min_vram,
                    )
                    if free_mb >= min_vram:
                        dev = f"cuda:{i}"
                        engine = create_engine(
                            backend=self.config.backend,
                            device=dev,
                            img_size=self.config.img_size,
                        )
                        self.engines.append((dev, engine))
                        logger.info("Engine loaded on %s", dev)
                    else:
                        logger.warning(
                            "GPU %d (%s): skipped (%.0f MB free < %d MB needed)",
                            i, name, free_mb, min_vram,
                        )

                if not self.engines:
                    logger.info("No GPU with sufficient VRAM. Falling back to CPU.")
                    devices = ["cpu"]
                else:
                    return  # Engines already loaded

        logger.info("Loading engines on devices: %s", devices)
        for dev in devices:
            engine = create_engine(
                backend=self.config.backend,
                device=dev,
                img_size=self.config.img_size,
            )
            self.engines.append((dev, engine))
            logger.info("Engine loaded on %s", dev)

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
        prefetch = self.config.prefetch_depth or (num_gpus * 4)
        num_readers = self.config.read_workers or max(2, os.cpu_count() // 2)

        # Queues
        work_q: queue.Queue[FramePacket | object] = queue.Queue(maxsize=prefetch)
        write_q: queue.Queue[ResultPacket | object] = queue.Queue()

        # Tracking
        completed_count = [0]
        failed_frames: list[int] = []
        failed_lock = threading.Lock()

        self._shutdown_event.clear()

        # --- Progress bar ---
        from tqdm import tqdm

        pbar = tqdm(total=num_frames, desc="CorridorKey", unit="frame", dynamic_ncols=True)

        # --- Reader stage ---
        def reader_task():
            """Submit frame reads to thread pool, feed work_q."""
            with ThreadPoolExecutor(max_workers=num_readers) as pool:
                futures = []
                for i in range(num_frames):
                    if self._shutdown_event.is_set():
                        break
                    f = pool.submit(
                        _read_frame_pair,
                        input_paths[i],
                        alpha_paths[i],
                        i,
                        input_stems[i],
                        settings.input_is_linear,
                    )
                    futures.append((i, f))

                for i, f in futures:
                    if self._shutdown_event.is_set():
                        break
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

        # --- Inference stage (one thread per GPU, work-stealing) ---
        def inference_worker(device_str: str, engine: Any):
            """Pull frames from work_q, run inference, push to write_q."""
            if device_str.startswith("cuda"):
                dev_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                torch.cuda.set_device(dev_idx)

            while not self._shutdown_event.is_set():
                item = work_q.get()
                if item is _SHUTDOWN:
                    break

                packet: FramePacket = item
                try:
                    result = engine.process_frame(
                        packet.img_srgb,
                        packet.mask_linear,
                        input_is_linear=settings.input_is_linear,
                        fg_is_straight=True,
                        despill_strength=settings.despill_strength,
                        auto_despeckle=settings.auto_despeckle,
                        despeckle_size=settings.despeckle_size,
                        refiner_scale=settings.refiner_scale,
                    )
                    write_q.put(ResultPacket(
                        index=packet.index,
                        input_stem=packet.input_stem,
                        alpha=result["alpha"],
                        fg=result["fg"],
                        comp=result["comp"],
                        processed=result.get("processed"),
                    ))
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "\033[1mCUDA OOM on %s for frame %d — taking GPU offline\033[0m",
                        device_str, packet.index,
                    )
                    torch.cuda.empty_cache()
                    with failed_lock:
                        failed_frames.append(packet.index)
                    break  # Take this GPU offline
                except Exception as e:
                    logger.warning("Inference error frame %d on %s: %s", packet.index, device_str, e)
                    with failed_lock:
                        failed_frames.append(packet.index)

        # --- Writer stage (single thread, serialized) ---
        def writer_task():
            """Consume results and write to disk. Updates progress bar."""
            active_inference = num_gpus

            while active_inference > 0 or not write_q.empty():
                try:
                    item = write_q.get(timeout=0.5)
                except queue.Empty:
                    if self._shutdown_event.is_set():
                        break
                    continue

                if item is _SHUTDOWN:
                    active_inference -= 1
                    continue

                result: ResultPacket = item
                try:
                    _write_result(
                        result,
                        output_dirs["fg"],
                        output_dirs["matte"],
                        output_dirs["comp"],
                        output_dirs["processed"],
                    )
                    completed_count[0] += 1
                    pbar.update(1)
                except Exception as e:
                    logger.warning("Write error frame %d: %s", result.index, e)
                    with failed_lock:
                        failed_frames.append(result.index)
                    pbar.update(1)

        # --- Launch pipeline ---
        reader_thread = threading.Thread(target=reader_task, name="pipeline-reader", daemon=True)
        writer_thread = threading.Thread(target=writer_task, name="pipeline-writer", daemon=True)

        inference_threads = []
        for device_str, engine in self.engines:
            t = threading.Thread(
                target=inference_worker,
                args=(device_str, engine),
                name=f"pipeline-infer-{device_str}",
                daemon=True,
            )
            inference_threads.append(t)

        try:
            reader_thread.start()
            for t in inference_threads:
                t.start()
            writer_thread.start()

            # Wait for reader to finish submitting
            reader_thread.join()

            # Wait for all inference threads
            for t in inference_threads:
                t.join()

            # Signal writer to finish
            for _ in range(1):
                write_q.put(_SHUTDOWN)

            writer_thread.join()

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted — shutting down...")
            self._shutdown_event.set()
            # Drain queues to unblock threads
            while not work_q.empty():
                try:
                    work_q.get_nowait()
                except queue.Empty:
                    break
            reader_thread.join(timeout=5)
            for t in inference_threads:
                t.join(timeout=5)
            writer_thread.join(timeout=5)

        finally:
            pbar.close()

        # --- Summary ---
        num_failed = len(failed_frames)
        if num_failed > 0:
            failed_frames.sort()
            logger.warning(
                "\033[1m%d frame(s) failed: %s\033[0m",
                num_failed,
                failed_frames[:20] if num_failed > 20 else failed_frames,
            )

        return {
            "total": num_frames,
            "completed": completed_count[0],
            "failed": num_failed,
            "skipped_frames": sorted(failed_frames),
        }

    def shutdown(self) -> None:
        """Signal all threads to stop."""
        self._shutdown_event.set()
