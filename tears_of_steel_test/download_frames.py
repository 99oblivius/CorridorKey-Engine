"""Download first 100 EXR frames from Tears of Steel footage.

Source: https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/02_3c/linear/
Format: OpenEXR 16-bit half-float, 4096x2160, linear color space
License: CC-BY 3.0 (c) Blender Foundation | mango.blender.org

Usage:
    python tears_of_steel_test/download_frames.py
"""

import os
import sys
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/02_3c/linear/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "frames")
NUM_FRAMES = 100
MAX_WORKERS = 4  # concurrent downloads


def download_frame(frame_idx: int) -> tuple[int, bool, str]:
    filename = f"02_3c_{frame_idx:05d}.exr"
    url = BASE_URL + filename
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if size_mb > 10:  # sanity check: real EXR files are ~51MB
            return frame_idx, True, f"already exists ({size_mb:.1f} MB)"

    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        return frame_idx, True, f"downloaded ({size_mb:.1f} MB)"
    except Exception as e:
        return frame_idx, False, str(e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check how many already exist
    existing = sum(
        1 for i in range(NUM_FRAMES)
        if os.path.exists(os.path.join(OUTPUT_DIR, f"02_3c_{i:05d}.exr"))
    )
    if existing == NUM_FRAMES:
        print(f"All {NUM_FRAMES} frames already downloaded in {OUTPUT_DIR}")
        return

    print(f"Downloading {NUM_FRAMES} EXR frames from Tears of Steel (02_3c)")
    print(f"  Source: {BASE_URL}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Already have: {existing}/{NUM_FRAMES}")
    print(f"  Expected total size: ~{NUM_FRAMES * 51 / 1024:.1f} GB")
    print(f"  Workers: {MAX_WORKERS}")
    print()

    start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_frame, i): i
            for i in range(NUM_FRAMES)
        }
        for future in as_completed(futures):
            idx, success, msg = future.result()
            completed += 1
            if not success:
                failed += 1
                print(f"  FAILED [{completed}/{NUM_FRAMES}] frame {idx:05d}: {msg}")
            else:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (NUM_FRAMES - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{NUM_FRAMES}] frame {idx:05d}: {msg}  "
                      f"({rate:.1f} frames/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Downloaded: {completed - failed}, Failed: {failed}")


if __name__ == "__main__":
    main()
