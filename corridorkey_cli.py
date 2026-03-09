"""CorridorKey command-line interface and interactive wizard.

This module handles CLI argument parsing, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage (via launcher scripts):
    uv run python corridorkey_cli.py --action wizard --win_path "V:\\..."
    uv run python corridorkey_cli.py --action run_inference
    uv run python corridorkey_cli.py --action generate_alphas
    uv run python corridorkey_cli.py --action list
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
import warnings

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    generate_alphas,
    generate_alphas_birefnet,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
    scan_clips,
)
from device_utils import resolve_device

logger = logging.getLogger(__name__)


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use.

    This is called once at startup when running from the command line.
    It is NOT called when importing clip_manager as a library.
    """
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def interactive_wizard(win_path: str, device: str | None = None) -> None:
    print("\n" + "=" * 60)
    print(" CORRIDOR KEY - SMART WIZARD")
    print("=" * 60)

    # 1. Resolve Path
    print(f"Windows Path: {win_path}")

    # Check if we are running locally where the Windows path exists
    if os.path.exists(win_path):
        process_path = win_path
        print(f"Running locally dynamically using path: {process_path}")
    else:
        # Fallback to mapping for remote linux execution
        process_path = map_path(win_path)
        print(f"Linux/Remote Path:   {process_path}")

        if not os.path.exists(process_path):
            print("\n[ERROR] Path does not exist locally OR on Linux mount!")
            print(f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}")
            return

    # 2. Analyze
    # We treat process_path as the ROOT containing SHOTS
    # Or is process_path the SHOT itself?
    # HEURISTIC: If it contains "Input", it's a shot. If it contains folders, it's a project.

    # Let's assume it's a folder containing CLIPS (Batch Mode)
    # But if the user drops it IN a shot folder, we should handle that too.

    target_is_shot = False
    if os.path.exists(os.path.join(process_path, "Input")) or glob.glob(os.path.join(process_path, "Input.*")):
        target_is_shot = True

    work_dirs = []
    if target_is_shot:
        work_dirs = [process_path]
    else:
        # Scan subfolders
        work_dirs = [
            os.path.join(process_path, d)
            for d in os.listdir(process_path)
            if os.path.isdir(os.path.join(process_path, d))
        ]
        # Filter out output/hints
        work_dirs = [
            d
            for d in work_dirs
            if os.path.basename(d) not in ["Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"]
        ]

    print(f"\nFound {len(work_dirs)} potential clip folders.")

    # Check for loose videos in root
    loose_videos = [
        f for f in os.listdir(process_path) if is_video_file(f) and os.path.isfile(os.path.join(process_path, f))
    ]

    # Check if existing folders need organization
    dirs_needing_org = []
    for d in work_dirs:
        # Check for Input
        has_input = os.path.exists(os.path.join(d, "Input")) or glob.glob(os.path.join(d, "Input.*"))
        # Check for hints
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))

        if not has_input or not has_alpha or not has_mask:
            dirs_needing_org.append(d)

    if loose_videos or dirs_needing_org:
        if loose_videos:
            print(f"Found {len(loose_videos)} loose video files that need organization:")
            for v in loose_videos:
                print(f"  - {v}")

        if dirs_needing_org:
            print(f"Found {len(dirs_needing_org)} folders that might need setup (Hints/Input):")
            # Limit output if too many
            if len(dirs_needing_org) < 10:
                for d in dirs_needing_org:
                    print(f"  - {os.path.basename(d)}")
            else:
                print(f"  - ...and {len(dirs_needing_org)} others.")

        # 3. Organize Loop
        yn = input("\n[1] Organize Clips & Create Hint Folders? [y/N]: ").strip().lower()
        if yn == "y":
            # Organize loose videos first
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(process_path, clip_name)

                if os.path.exists(target_folder):
                    logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(process_path, v), target_file)
                    logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")

                    # Also initialize hints immediately
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to organize video '{v}': {e}")

            # Organize existing folders
            for d in work_dirs:
                organize_target(d)
            print("Organization Complete.")

            # Re-scan in case structure changed
            # If it was a shot, it's still a shot (unless we messed it up)
            # If it wasn't a shot, we re-scan subdirs
            if not target_is_shot:
                work_dirs = [
                    os.path.join(process_path, d)
                    for d in os.listdir(process_path)
                    if os.path.isdir(os.path.join(process_path, d))
                ]
                work_dirs = [
                    d for d in work_dirs if os.path.basename(d) not in ["Output", "AlphaHint", "VideoMamaMaskHint"]
                ]

    # 4. Status Check Loop
    while True:
        ready = []
        masked = []
        raw = []

        for d in work_dirs:
            entry = ClipEntry(os.path.basename(d), d)
            try:
                entry.find_assets()  # This checks Input and AlphaHint
            except (FileNotFoundError, ValueError, OSError):
                pass  # Might act up if Input missing

            # Check VideoMamaMaskHint (Strict: videomamamaskhint.ext or VideoMamaMaskHint/)
            has_mask = False
            mask_dir = os.path.join(d, "VideoMamaMaskHint")

            # 1. Directory Check
            if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                has_mask = True

            # 2. File Check (Strict Stem Match)
            if not has_mask:
                for f in os.listdir(d):
                    stem, _ = os.path.splitext(f)
                    if stem.lower() == "videomamamaskhint" and is_video_file(f):
                        has_mask = True
                        break

            if entry.alpha_asset:
                ready.append(entry)
            elif has_mask:
                masked.append(entry)
            else:
                raw.append(entry)

        print("\n" + "-" * 40)
        print("STATUS REPORT:")
        print(f"  READY (AlphaHint found): {len(ready)}")
        for c in ready:
            print(f"    - {c.name}")

        print(f"  MASKED (VideoMamaMaskHint found): {len(masked)}")
        for c in masked:
            print(f"    - {c.name}")

        print(f"  RAW (Input only):        {len(raw)}")
        for c in raw:
            print(f"    - {c.name}")
        print("-" * 40 + "\n")

        # Combine checks for actions
        missing_alpha = masked + raw

        print("\nACTIONS:")
        if missing_alpha:
            print(f"  [v] Run VideoMaMa (Found {len(masked)} ready with masks)")
            print(f"  [g] Run GVM (Auto-Matte on {len(raw)} clips without Mask Hint)")
            print(f"  [b] Run BiRefNet (Lightweight auto-matte on {len(raw)} clips, ~4GB VRAM)")

        if ready:
            print(f"  [i] Run Inference (on {len(ready)} ready clips)")

        print("  [r] Re-Scan Folders")
        print("  [q] Quit")

        choice = input("\nSelect Action: ").strip().lower()

        if choice == "v":
            # VideoMaMa
            print("\n--- VideoMaMa ---")
            print("Scanning for VideoMamaMaskHints...")
            # We pass ALL missing alpha clips. run_videomama checks for the actual files.
            run_videomama(missing_alpha, chunk_size=50, device=device)
            input("VideoMaMa batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "g":
            # GVM
            print("\n--- GVM Auto-Matte ---")
            print(f"This will generate alphas for {len(raw)} clips that have NO Mask Hint.")

            yn = input("Proceed with GVM? [y/N]: ").strip().lower()
            if yn == "y":
                generate_alphas(raw, device=device)
                input("GVM batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "b":
            # BiRefNet
            print("\n--- BiRefNet Auto-Matte ---")
            print(f"This will generate alphas for {len(raw)} clips using BiRefNet (~4GB VRAM).")

            yn = input("Proceed with BiRefNet? [y/N]: ").strip().lower()
            if yn == "y":
                generate_alphas_birefnet(raw, device=device)
                input("BiRefNet batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "i":
            # Inference
            print("\n--- Corridor Key Inference ---")
            try:
                run_inference(ready, device=device)
            except (RuntimeError, FileNotFoundError) as e:
                logger.error(f"Inference failed: {e}")
            input("Inference batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "r":
            print("\nRe-scanning...")
            continue

        elif choice == "q":
            break

        else:
            print("Invalid selection.")
            continue

    print("\nWizard Complete. Goodbye!")


def main() -> None:
    _configure_environment()

    parser = argparse.ArgumentParser(description="CorridorKey Clip Manager")
    parser.add_argument("--action", choices=["generate_alphas", "generate_alphas_birefnet", "run_inference", "list", "wizard"], required=True)
    parser.add_argument("--win_path", help=r"Windows Path (example: V:\...) for Wizard Mode", default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "torch_optimized", "mlx"],
        default="auto",
        help="Inference backend (default: auto). torch_optimized uses tiled "
        "refiner and memory hygiene to reduce VRAM from ~22.7GB to <8GB.",
    )

    # --- Optimization profile & per-optimization flags ---
    parser.add_argument(
        "--profile",
        choices=["original", "optimized", "experimental"],
        default=None,
        help="Optimization profile (default: inferred from --backend). "
        "Per-optimization flags below override profile settings.",
    )
    parser.add_argument("--flash-attention", action="store_true", default=None, help="Enable FlashAttention patching")
    parser.add_argument("--no-flash-attention", dest="flash_attention", action="store_false")
    parser.add_argument("--tiled-refiner", action="store_true", default=None, help="Enable tiled CNN refiner")
    parser.add_argument("--no-tiled-refiner", dest="tiled_refiner", action="store_false")
    parser.add_argument("--cache-clearing", action="store_true", default=None, help="Enable CUDA cache clearing")
    parser.add_argument("--no-cache-clearing", dest="cache_clearing", action="store_false")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true", default=None, help="Disable cuDNN benchmark")
    parser.add_argument("--no-disable-cudnn-benchmark", dest="disable_cudnn_benchmark", action="store_false")
    parser.add_argument("--token-routing", action="store_true", default=None, help="Enable experimental token routing")
    parser.add_argument("--no-token-routing", dest="token_routing", action="store_false")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for tiled refiner (default: 512)")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Tile overlap in pixels (default: 128)")
    parser.add_argument("--metrics", action="store_true", default=False, help="Enable per-stage performance metrics")

    args = parser.parse_args()

    device = resolve_device(args.device)
    backend = args.backend
    logger.info(f"Using device: {device}, backend: {backend}")

    # Build OptimizationConfig from profile + overrides
    optimization_config = _build_optimization_config(args)
    if optimization_config is not None:
        logger.info(optimization_config.summary())

    try:
        if args.action == "list":
            scan_clips()
        elif args.action == "generate_alphas":
            clips = scan_clips()
            generate_alphas(clips, device=device)
        elif args.action == "generate_alphas_birefnet":
            clips = scan_clips()
            generate_alphas_birefnet(clips, device=device)
        elif args.action == "run_inference":
            clips = scan_clips()
            run_inference(clips, device=device, backend=backend, optimization_config=optimization_config)
        elif args.action == "wizard":
            if not args.win_path:
                print("Error: --win_path required for wizard.")
            else:
                interactive_wizard(args.win_path, device=device)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


def _build_optimization_config(args):
    """Build an OptimizationConfig from CLI args.

    Returns None if no profile or optimization flags were specified
    (let the backend factory pick its own default).
    """
    from dataclasses import replace

    from CorridorKeyModule.optimization_config import OptimizationConfig

    # Determine base config from profile or backend
    if args.profile:
        config = OptimizationConfig.from_profile(args.profile)
    elif args.backend == "torch_optimized":
        config = OptimizationConfig.optimized()
    elif args.backend == "torch":
        config = OptimizationConfig.original()
    else:
        # Check if any override flags were set
        has_overrides = (
            any(
                getattr(args, attr, None) is not None
                for attr in (
                    "flash_attention",
                    "tiled_refiner",
                    "cache_clearing",
                    "disable_cudnn_benchmark",
                    "token_routing",
                    "tile_size",
                    "tile_overlap",
                )
            )
            or args.metrics
        )
        if not has_overrides:
            return None
        config = OptimizationConfig()

    # Apply per-flag overrides
    overrides = {}
    if args.flash_attention is not None:
        overrides["flash_attention"] = args.flash_attention
    if args.tiled_refiner is not None:
        overrides["tiled_refiner"] = args.tiled_refiner
    if args.cache_clearing is not None:
        overrides["cache_clearing"] = args.cache_clearing
    if args.disable_cudnn_benchmark is not None:
        overrides["disable_cudnn_benchmark"] = args.disable_cudnn_benchmark
    if args.token_routing is not None:
        overrides["token_routing"] = args.token_routing
    if args.tile_size is not None:
        overrides["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        overrides["tile_overlap"] = args.tile_overlap
    if args.metrics:
        overrides["enable_metrics"] = True

    if overrides:
        config = replace(config, **overrides)

    return config


if __name__ == "__main__":
    main()
