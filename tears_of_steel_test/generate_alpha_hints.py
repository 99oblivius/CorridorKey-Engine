"""Generate coarse alpha hints for Tears of Steel green screen EXR footage.

Uses HSV-based chroma keying to detect the green screen background, then
produces soft, slightly eroded masks suitable as input hints for CorridorKey.

The EXR frames are in linear color space, so we convert to sRGB before HSV
analysis for more perceptually accurate green detection.

Usage:
    uv run python tears_of_steel_test/generate_alpha_hints.py
"""

import os
import sys
import glob
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "frames")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "alpha_hints")

# HSV thresholds for green screen detection (in sRGB/HSV space)
# Hue: 35-85 covers green range (OpenCV uses 0-180 scale)
# Saturation: >40 to avoid desaturated greens / neutral colors
# Value: >30 to avoid very dark areas
GREEN_HUE_LOW = 35
GREEN_HUE_HIGH = 85
GREEN_SAT_MIN = 40
GREEN_VAL_MIN = 30

# Post-processing: erosion shrinks the foreground slightly (model prefers
# under-prediction), blur makes edges soft/coarse as the model expects
ERODE_KERNEL_SIZE = 7
BLUR_KERNEL_SIZE = 21


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear float image to sRGB float image (piecewise transfer)."""
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def generate_alpha_hint(frame_linear: np.ndarray) -> np.ndarray:
    """Generate a coarse alpha hint from a linear-space green screen frame.

    Returns a float32 [H, W] mask where 1.0 = foreground, 0.0 = background.
    """
    # Convert linear EXR to sRGB for perceptually accurate HSV analysis
    frame_srgb = linear_to_srgb(frame_linear)

    # Convert to uint8 for OpenCV HSV conversion
    frame_u8 = (frame_srgb * 255.0).astype(np.uint8)

    # BGR->HSV (OpenCV expects BGR, our EXR was read as BGR by cv2.imread)
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2HSV)

    # Detect green screen pixels
    green_mask = cv2.inRange(
        hsv,
        (GREEN_HUE_LOW, GREEN_SAT_MIN, GREEN_VAL_MIN),
        (GREEN_HUE_HIGH, 255, 255),
    )

    # Invert: green = background (0), non-green = foreground (1)
    fg_mask = cv2.bitwise_not(green_mask)

    # Erode slightly to under-predict foreground (model handles this better)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE)
    )
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)

    # Gaussian blur to create soft/coarse edges
    fg_mask = cv2.GaussianBlur(fg_mask, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    # Normalize to float32 [0, 1]
    return fg_mask.astype(np.float32) / 255.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exr_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.exr")))
    if not exr_files:
        print(f"No EXR frames found in {FRAMES_DIR}")
        print("Run download_frames.py first.")
        sys.exit(1)

    # Check which hints already exist
    existing = 0
    to_process = []
    for exr_path in exr_files:
        basename = os.path.splitext(os.path.basename(exr_path))[0]
        hint_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
        if os.path.exists(hint_path):
            existing += 1
        else:
            to_process.append((exr_path, hint_path))

    print(f"Alpha hint generation (HSV chroma key)")
    print(f"  Frames dir: {FRAMES_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Total EXR frames: {len(exr_files)}")
    print(f"  Already generated: {existing}")
    print(f"  To process: {len(to_process)}")
    print()

    if not to_process:
        print("All alpha hints already generated.")
        return

    start = time.time()
    for i, (exr_path, hint_path) in enumerate(to_process):
        frame = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"  WARNING: Failed to read {exr_path}, skipping")
            continue

        # Ensure float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        alpha_hint = generate_alpha_hint(frame)

        # Save as uint8 PNG (standard format for alpha hints)
        hint_u8 = (np.clip(alpha_hint, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(hint_path, hint_u8)

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {rate:.1f} frames/s, ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Generated {len(to_process)} alpha hints.")


if __name__ == "__main__":
    main()
