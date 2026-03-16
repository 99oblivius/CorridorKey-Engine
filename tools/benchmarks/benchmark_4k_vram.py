"""4K Benchmark: Compare all optimization profiles.

Processes real 4K green-screen footage (Tears of Steel, 4096x2160 EXR frames
in linear color space) through up to four optimization profiles:
  1. Original     (fp32, no optimizations, no compilation)
  2. Optimized    (fp16, flash+tiled+cache, no compilation)
  3. Experimental (fp16, torch.compile reduce-overhead, token routing)
  4. Performance  (fp16, max-autotune, full refiner, cuDNN benchmarking)

Each profile runs in its own subprocess for clean GPU state.  Outputs comp
and alpha EXR sequences for each profile so quality can be compared.

Source footage:
  Tears of Steel (CC-BY 3.0) (c) Blender Foundation | mango.blender.org
  https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/02_3c/linear/

Usage:
    uv run python tools/benchmarks/benchmark_4k_vram.py
    uv run python tools/benchmarks/benchmark_4k_vram.py --profiles original optimized
    uv run python tools/benchmarks/benchmark_4k_vram.py --frames 50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRAMES_DIR = os.path.join("benchmark", "frames")
HINT_DIR = os.path.join("benchmark", "alpha_hints")
OUTPUT_DIR = "Output"
REPORT_PATH = os.path.join("docs", "BENCHMARK_RESULTS.md")
DEFAULT_NUM_FRAMES = 100  # first 100 frames from the sequence

CONFIGS: list[tuple[str, str]] = [
    ("Original (fp32, no optimizations)", "original"),
    ("Optimized (fp16, flash+tiled+cache)", "optimized"),
    ("Experimental (fp16, torch.compile reduce-overhead)", "experimental"),
    ("Performance (fp16, max-autotune, full refiner)", "performance"),
]

# ---------------------------------------------------------------------------
# Worker script (runs in subprocess)
# ---------------------------------------------------------------------------

WORKER_SCRIPT = r'''
"""Worker: process 4K EXR frame sequence, write output EXRs, report metrics."""
import json, sys, time, os, glob, contextlib, io, threading

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU memory poller -- samples actual device memory
# ---------------------------------------------------------------------------
class GPUMemoryPoller:
    """Background thread that polls torch.cuda.mem_get_info() to track
    the real peak GPU memory usage, including cuDNN workspace and other
    allocations outside PyTorch's caching allocator."""

    def __init__(self, device=0, interval_ms=50):
        self.device = device
        self.interval = interval_ms / 1000.0
        self._peak_used_bytes = 0
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        # Get total GPU memory once
        _, self.total_bytes = torch.cuda.mem_get_info(self.device)

    def _poll_loop(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self.device)
            used = total - free
            if used > self._peak_used_bytes:
                self._peak_used_bytes = used
            self._samples.append(used)
            self._stop.wait(self.interval)

    def start(self):
        self._peak_used_bytes = 0
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset_peak(self):
        """Reset peak for per-frame tracking."""
        self._peak_used_bytes = 0

    @property
    def peak_used_mb(self):
        return self._peak_used_bytes / (1024**2)

    @property
    def current_used_mb(self):
        free, total = torch.cuda.mem_get_info(self.device)
        return (total - free) / (1024**2)

    @property
    def total_mb(self):
        return self.total_bytes / (1024**2)

profile_name = sys.argv[1]
frames_dir   = sys.argv[2]
output_dir   = sys.argv[3]
tag          = sys.argv[4]
hint_dir     = sys.argv[5]
num_frames   = int(sys.argv[6])

# ---- Build config via profile ----
from CorridorKeyModule.optimization_config import OptimizationConfig
config = OptimizationConfig.from_profile(profile_name)

from CorridorKeyModule.engine_factory import create_engine

# ---- Discover EXR frames and alpha hints ----
exr_files = sorted(glob.glob(os.path.join(frames_dir, "*.exr")))[:num_frames]
hint_files = sorted([f for f in os.listdir(hint_dir) if f.lower().endswith(('.png', '.exr', '.jpg', '.tif', '.tiff'))]) if os.path.isdir(hint_dir) else []
total_frames = len(exr_files)

if total_frames == 0:
    print("ERROR: No EXR frames found", flush=True)
    sys.exit(1)

# Read first frame to get dimensions
first_frame = cv2.imread(exr_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
h, w = first_frame.shape[:2]
fps = 24.0  # Tears of Steel footage is 24 fps

print(f"Found {total_frames} EXR frames ({w}x{h}) and {len(hint_files)} alpha hints", flush=True)
print(f"Profile: {profile_name} | Config: {config.summary()}", flush=True)

result = {
    "resolution": f"{w}x{h}",
    "total_frames": total_frames,
    "fps": fps,
    "config_summary": config.summary(),
    "active_opts": config.active_optimizations(),
    "model_precision": config.model_precision,
    "compile_mode": config.compile_mode,
    "tag": tag,
}

# ---- Load engine ----
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

load_start = time.perf_counter()
engine = create_engine(
    device="cuda",
    img_size=2048,
    optimization_config=config,
)
load_time_ms = (time.perf_counter() - load_start) * 1000
model_vram_mb = torch.cuda.memory_allocated() / (1024**2)

result["model_vram_mb"] = round(model_vram_mb, 1)
result["load_time_ms"] = round(load_time_ms, 1)

# ---- Start GPU memory poller ----
poller = GPUMemoryPoller(device=0, interval_ms=25)
idle_gpu_mb = poller.current_used_mb
result["idle_gpu_mb"] = round(idle_gpu_mb, 1)
print(f"Device memory after model load: {idle_gpu_mb:.0f} MB (this is the idle baseline)", flush=True)
poller.start()

# ---- Prepare output directories ----
comp_dir  = os.path.join(output_dir, f"comp_{tag}")
alpha_dir = os.path.join(output_dir, f"alpha_{tag}")
os.makedirs(comp_dir, exist_ok=True)
os.makedirs(alpha_dir, exist_ok=True)

# EXR write params: PXR24 compression, half-float
exr_params = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]

# ---- Process frames ----
torch.cuda.reset_peak_memory_stats()

frame_times = []
vram_peaks_per_frame = []
device_peaks_per_frame = []
overall_start = time.perf_counter()

for frame_idx, exr_path in enumerate(exr_files):
    # Read EXR frame (linear float, BGR)
    frame_bgr = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if frame_bgr is None:
        print(f"  WARNING: Failed to read {exr_path}, skipping", flush=True)
        continue

    # Convert BGR to RGB, ensure float32
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Load alpha hint for this frame
    if frame_idx < len(hint_files):
        hint_path = os.path.join(hint_dir, hint_files[frame_idx])
        mask_raw = cv2.imread(hint_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            print(f"  WARNING: Failed to read hint {hint_path}, using zeros", flush=True)
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            # Handle multi-channel (take first channel)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            # Normalize to 0-1
            if mask_raw.dtype == np.uint8:
                mask = mask_raw.astype(np.float32) / 255.0
            elif mask_raw.dtype == np.uint16:
                mask = mask_raw.astype(np.float32) / 65535.0
            else:
                mask = mask_raw.astype(np.float32)
            # Resize to match frame dimensions if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"  WARNING: No hint for frame {frame_idx}, using zeros", flush=True)
        mask = np.zeros((h, w), dtype=np.float32)

    torch.cuda.reset_peak_memory_stats()
    poller.reset_peak()
    t0 = time.perf_counter()

    # Suppress per-frame print output from engine
    with contextlib.redirect_stdout(io.StringIO()):
        output = engine.process_frame(frame_rgb, mask, input_is_linear=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    device_peak_mb = poller.peak_used_mb

    frame_times.append(elapsed_ms)
    vram_peaks_per_frame.append(peak_mb)
    device_peaks_per_frame.append(device_peak_mb)

    # Write processed RGBA EXR (linear premultiplied - ready for compositing)
    processed = output["processed"]  # [H, W, 4] linear float RGBA
    # Convert RGBA to BGRA for OpenCV
    processed_bgra = cv2.cvtColor(processed, cv2.COLOR_RGBA2BGRA)
    basename = os.path.splitext(os.path.basename(exr_path))[0]
    cv2.imwrite(os.path.join(comp_dir, f"{basename}.exr"), processed_bgra, exr_params)

    # Write alpha channel as single-channel EXR
    alpha = output["alpha"]
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    cv2.imwrite(os.path.join(alpha_dir, f"{basename}.exr"), alpha, exr_params)

    frame_idx_display = frame_idx + 1
    # Progress every 10 frames
    if frame_idx_display % 10 == 0 or frame_idx_display == total_frames:
        avg_ms = sum(frame_times) / len(frame_times)
        eta_s = (total_frames - frame_idx_display) * avg_ms / 1000
        print(f"  [{frame_idx_display}/{total_frames}] "
              f"last={elapsed_ms:.0f}ms avg={avg_ms:.0f}ms "
              f"device_peak={device_peak_mb:.0f}MB ETA={eta_s:.0f}s",
              flush=True)

overall_elapsed_s = time.perf_counter() - overall_start
poller.stop()

# ---- Collect final metrics ----
overall_peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
overall_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

ft = frame_times
result["status"] = "OK"
result["overall_time_s"] = round(overall_elapsed_s, 2)
result["frames_processed"] = len(frame_times)
result["avg_frame_ms"] = round(sum(ft) / len(ft), 1)
result["min_frame_ms"] = round(min(ft), 1)
result["max_frame_ms"] = round(max(ft), 1)
result["median_frame_ms"] = round(sorted(ft)[len(ft)//2], 1)
result["effective_fps"] = round(len(frame_times) / overall_elapsed_s, 2)
result["peak_vram_allocated_mb"] = round(overall_peak_alloc, 1)
result["peak_vram_reserved_mb"] = round(overall_peak_reserved, 1)
result["avg_vram_peak_per_frame_mb"] = round(sum(vram_peaks_per_frame) / len(vram_peaks_per_frame), 1)
result["max_vram_peak_per_frame_mb"] = round(max(vram_peaks_per_frame), 1)

# Device-level GPU memory (actual usage including cuDNN, CUDA context, etc.)
result["device_peak_gpu_mb"] = round(max(device_peaks_per_frame), 1)
result["device_avg_peak_gpu_mb"] = round(sum(device_peaks_per_frame) / len(device_peaks_per_frame), 1)
result["device_idle_gpu_mb"] = round(idle_gpu_mb, 1)
result["device_total_gpu_mb"] = round(poller.total_mb, 1)

result["comp_output"] = comp_dir
result["alpha_output"] = alpha_dir

# First 5 frame times (warmup check) vs last 5
if len(ft) >= 10:
    result["first5_avg_ms"] = round(sum(ft[:5]) / 5, 1)
    result["last5_avg_ms"] = round(sum(ft[-5:]) / 5, 1)

print("===RESULT_JSON===")
print(json.dumps(result))
'''

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_config(label: str, profile_name: str, tag: str, num_frames: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"  Profile: {profile_name}")
    print(f"  Processing {num_frames} 4K EXR frames (Tears of Steel)")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-c", WORKER_SCRIPT,
        profile_name, FRAMES_DIR, OUTPUT_DIR, tag, HINT_DIR, str(num_frames),
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall_time = time.time() - start

    if proc.stdout:
        for line in proc.stdout.splitlines():
            if not line.startswith("===RESULT_JSON==="):
                print(line)

    if proc.stderr:
        for line in proc.stderr.splitlines():
            if line.strip():
                print(f"  [stderr] {line}")

    result = {"label": label, "wall_time_s": round(wall_time, 1)}

    if "===RESULT_JSON===" in proc.stdout:
        json_str = proc.stdout.split("===RESULT_JSON===\n", 1)[1].strip()
        try:
            result.update(json.loads(json_str))
        except json.JSONDecodeError:
            result["status"] = "PARSE_ERROR"
            result["error"] = json_str[:200]
    else:
        result["status"] = "CRASH"
        result["error"] = (proc.stderr or proc.stdout or "No output")[:300]

    status = result.get("status", "UNKNOWN")
    if status == "OK":
        fps = result.get("effective_fps", 0)
        device_peak = result.get("device_peak_gpu_mb", 0)
        total = result.get("overall_time_s", 0)
        print(f"\n  DONE: {result.get('frames_processed',0)} frames in {total:.1f}s "
              f"({fps:.2f} fps) | Device peak: {device_peak:.0f} MB")
    else:
        print(f"\n  {status}: {result.get('error', 'unknown')[:100]}")

    return result


def generate_report(results: list[dict], gpu_info: str, num_frames: int) -> str:
    L: list[str] = []
    ok_results = [r for r in results if r.get("status") == "OK"]
    baseline = next((r for r in results if r.get("tag") == "original"), None)

    L.append("# CorridorKey 4K Benchmark -- Tears of Steel")
    L.append("")
    L.append("## Test Configuration")
    L.append("")
    L.append("- **Source footage**: Tears of Steel (scene 02_3c) -- CC-BY 3.0 (c) Blender Foundation | mango.blender.org")
    L.append("- **Format**: OpenEXR 16-bit half-float, linear color space")
    L.append("- **Resolution**: 4096x2160 (DCI 4K)")
    L.append(f"- **Frames**: {num_frames} (24 fps, ~{num_frames/24:.1f} seconds)")
    L.append("- **Model input size**: 2048x2048")
    L.append(f"- **GPU**: {gpu_info}")
    L.append("- **Alpha hints**: HSV chroma key (auto-generated from green screen footage)")
    L.append("- **Color pipeline**: `input_is_linear=True` -- engine handles linear-to-sRGB conversion internally")
    L.append("")

    # --- Head-to-head comparison ---
    if len(ok_results) >= 2:
        L.append("---")
        L.append("")
        L.append("## Head-to-Head Comparison")
        L.append("")

        # Build header row
        header_cells = ["Metric"]
        sep_cells = ["|---"]
        for r in ok_results:
            header_cells.append(r["label"])
            sep_cells.append("---:|")
        # Add delta column if we have a baseline
        if baseline and baseline.get("status") == "OK":
            header_cells.append("Best vs Original")
            sep_cells.append("---:|")

        L.append("| " + " | ".join(header_cells) + " |")
        L.append("|".join(sep_cells))

        def row(label: str, key: str, unit: str = "", fmt: str = ".1f") -> None:
            cells = [label]
            values = []
            for r in ok_results:
                v = r.get(key, 0)
                values.append(v)
                if isinstance(v, (int, float)):
                    cells.append(f"{v:{fmt}} {unit}")
                else:
                    cells.append(str(v))

            if baseline and baseline.get("status") == "OK":
                bv = baseline.get(key, 0)
                if isinstance(bv, (int, float)) and bv != 0:
                    # Find best non-baseline value
                    non_baseline = [v for v, r in zip(values, ok_results) if r.get("tag") != "original"]
                    if non_baseline:
                        # For time/ms metrics, lower is better; for fps, higher is better
                        is_higher_better = key in ("effective_fps",)
                        best = max(non_baseline) if is_higher_better else min(non_baseline)
                        delta = best - bv
                        pct = (delta / bv) * 100
                        sign = "+" if delta > 0 else ""
                        cells.append(f"{sign}{pct:.1f}%")
                    else:
                        cells.append("--")
                else:
                    cells.append("--")

            L.append("| " + " | ".join(cells) + " |")

        row("Total time", "overall_time_s", "s")
        row("Effective FPS", "effective_fps", "fps", fmt=".2f")
        row("Avg frame time", "avg_frame_ms", "ms")
        row("Median frame time", "median_frame_ms", "ms")
        row("Min frame time", "min_frame_ms", "ms")
        row("Max frame time", "max_frame_ms", "ms")

        L.append("")
        L.append("### GPU Memory")
        L.append("")

        header_cells2 = ["Metric"]
        sep_cells2 = ["|---"]
        for r in ok_results:
            header_cells2.append(r["label"])
            sep_cells2.append("---:|")
        if baseline and baseline.get("status") == "OK":
            header_cells2.append("Best vs Original")
            sep_cells2.append("---:|")

        L.append("| " + " | ".join(header_cells2) + " |")
        L.append("|".join(sep_cells2))

        row("Dedicated VRAM (physical)", "device_peak_gpu_mb", "MB")
        row("PyTorch allocator reserved", "peak_vram_reserved_mb", "MB")
        row("Idle after model load", "device_idle_gpu_mb", "MB")
        L.append("")

        # Warmup analysis
        has_warmup = any("first5_avg_ms" in r for r in ok_results)
        if has_warmup:
            L.append("### Warmup Effect (first 5 vs last 5 frames)")
            L.append("")
            L.append("| Profile | First 5 avg (ms) | Last 5 avg (ms) | Warmup overhead |")
            L.append("|---|---:|---:|---:|")
            for r in ok_results:
                if "first5_avg_ms" in r and "last5_avg_ms" in r:
                    f5, l5 = r["first5_avg_ms"], r["last5_avg_ms"]
                    wo = ((f5 - l5) / l5 * 100) if l5 else 0
                    L.append(f"| {r['label']} | {f5:.0f} | {l5:.0f} | {wo:+.1f}% |")
            L.append("")

    # --- Output files ---
    L.append("---")
    L.append("")
    L.append("## Output Files")
    L.append("")
    for r in ok_results:
        L.append(f"### {r['label']}")
        L.append("")
        L.append(f"- **Composite**: `{r.get('comp_output', 'N/A')}`")
        L.append(f"- **Alpha matte**: `{r.get('alpha_output', 'N/A')}`")
        L.append("")

    L.append("> Compare the composite and alpha EXR sequences side-by-side to evaluate quality")
    L.append("> differences between profiles. Output is linear premultiplied RGBA EXR -- ready")
    L.append("> for compositing in Nuke/Fusion/etc.")
    L.append("")

    # --- Detailed config info ---
    L.append("---")
    L.append("")
    L.append("## Configuration Details")
    L.append("")
    for r in ok_results:
        L.append(f"### {r['label']}")
        L.append("")
        L.append(f"- **Config**: {r.get('config_summary', 'N/A')}")
        L.append(f"- **Active optimizations**: {', '.join(r.get('active_opts', [])) or 'none'}")
        L.append(f"- **Model precision**: {r.get('model_precision', 'N/A')}")
        L.append(f"- **Compile mode**: {r.get('compile_mode', 'N/A')}")
        L.append(f"- **Model load time**: {r.get('load_time_ms', 0):.0f} ms")
        L.append(f"- **Frames processed**: {r.get('frames_processed', 0)}")
        L.append(f"- **Wall time (incl. subprocess)**: {r.get('wall_time_s', 0):.1f} s")
        L.append("")

    # --- Analysis ---
    L.append("---")
    L.append("")
    L.append("## Analysis")
    L.append("")

    if baseline and baseline.get("status") == "OK" and len(ok_results) >= 2:
        b_time = baseline.get("overall_time_s", 1)
        b_reserved = baseline.get("peak_vram_reserved_mb", 0)
        b_fps = baseline.get("effective_fps", 0)

        L.append(f"Processing {num_frames} frames of 4K EXR footage (Tears of Steel):")
        L.append("")

        for r in ok_results:
            if r.get("tag") == "original":
                continue
            o_time = r.get("overall_time_s", 1)
            o_reserved = r.get("peak_vram_reserved_mb", 0)
            o_fps = r.get("effective_fps", 0)
            speedup = ((b_time - o_time) / b_time * 100) if b_time else 0
            reserved_reduction = ((b_reserved - o_reserved) / b_reserved * 100) if b_reserved else 0

            L.append(f"**{r['label']}** vs Original:")
            L.append(f"- **Speed**: {o_fps:.2f} fps vs {b_fps:.2f} fps "
                     f"({'faster' if o_time < b_time else 'slower'} by {abs(speedup):.1f}%)")
            L.append(f"- **Reserved VRAM**: {o_reserved:.0f} MB vs {b_reserved:.0f} MB "
                     f"(**{'-' if reserved_reduction > 0 else '+'}{abs(reserved_reduction):.0f}%**)")
            L.append(f"- **Total time**: {o_time:.1f}s vs {b_time:.1f}s")
            L.append("")

    elif len(ok_results) == 1:
        r = ok_results[0]
        L.append(f"Single profile run ({r['label']}): {r.get('effective_fps', 0):.2f} fps, "
                 f"{r.get('overall_time_s', 0):.1f}s total, "
                 f"device peak {r.get('device_peak_gpu_mb', 0):.0f} MB.")
        L.append("")

    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CorridorKey 4K benchmark -- compare optimization profiles",
    )
    parser.add_argument(
        "--profiles", nargs="*", default=None,
        help="Profiles to benchmark (default: all). "
             "Valid: original, optimized, experimental, performance",
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames to process (default: {DEFAULT_NUM_FRAMES})",
    )
    args = parser.parse_args()
    num_frames = args.frames

    # Filter configs if --profiles specified
    if args.profiles is not None:
        valid_profiles = {profile for _, profile in CONFIGS}
        for p in args.profiles:
            if p not in valid_profiles:
                print(f"ERROR: Unknown profile '{p}'. Valid: {', '.join(sorted(valid_profiles))}")
                sys.exit(1)
        selected_configs = [(label, profile) for label, profile in CONFIGS if profile in args.profiles]
    else:
        selected_configs = list(CONFIGS)

    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleared {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(FRAMES_DIR):
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        print("Run: uv run python tools/benchmarks/download_frames.py")
        sys.exit(1)

    # Count available frames
    import glob as _glob
    available_frames = len(_glob.glob(os.path.join(FRAMES_DIR, "*.exr")))
    if available_frames < num_frames:
        print(f"WARNING: Only {available_frames} EXR frames available (need {num_frames})")
        print("Run: uv run python tools/benchmarks/download_frames.py")
    if not os.path.isdir(HINT_DIR):
        print(f"ERROR: Alpha hint directory not found: {HINT_DIR}")
        print("Run: uv run python tools/benchmarks/generate_alpha_hints.py")
        sys.exit(1)
    hint_count = len([f for f in os.listdir(HINT_DIR) if f.lower().endswith(('.png', '.exr', '.jpg'))])
    if hint_count == 0:
        print(f"ERROR: No alpha hints found in {HINT_DIR}")
        print("Run: uv run python tools/benchmarks/generate_alpha_hints.py")
        sys.exit(1)

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"{gpu_name} ({gpu_mem:.1f} GB)"
    except Exception:
        gpu_info = "Unknown"

    print("=" * 60)
    print("  CORRIDORKEY 4K BENCHMARK -- TEARS OF STEEL")
    print(f"  GPU: {gpu_info}")
    print(f"  Frames: {available_frames} EXR frames ({FRAMES_DIR})")
    print(f"  Alpha hints: {hint_count} ({HINT_DIR})")
    print(f"  Profiles: {', '.join(p for _, p in selected_configs)}")
    print("=" * 60)

    results = []
    for label, profile_name in selected_configs:
        result = run_config(label, profile_name, profile_name, num_frames)
        result["tag"] = profile_name
        results.append(result)

    # Ensure docs/ directory exists for report output
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    report = generate_report(results, gpu_info, num_frames)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Report: {REPORT_PATH}")
    print(f"{'='*60}\n")

    for r in results:
        if r.get("status") == "OK":
            print(f"  {r['label']}:")
            print(f"    {r.get('effective_fps',0):.2f} fps | "
                  f"{r.get('overall_time_s',0):.1f}s total | "
                  f"Device peak: {r.get('device_peak_gpu_mb',0):.0f} MB | "
                  f"Allocator reserved: {r.get('peak_vram_reserved_mb',0):.0f} MB")
            print(f"    Outputs: {r.get('comp_output','')}, {r.get('alpha_output','')}")


if __name__ == "__main__":
    main()
