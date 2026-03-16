#!/usr/bin/env python3
"""CorridorKey Engine — stdio client example.

Spawns the engine as a subprocess and communicates over stdin/stdout.
No socket setup, no daemon — just run this script.

Prerequisites:
    pip install corridorkey-engine   # or: uv sync
    # Download model weights (see README)

Usage:
    python docs/examples/stdio_client.py /path/to/project
"""

import sys

from ck_engine.api.events import (
    ClipStarted,
    JobCompleted,
    JobFailed,
    JobCancelled,
    JobProgress,
    ModelLoaded,
    ModelLoading,
    LogEvent,
)
from ck_engine.api.types import GenerateParams, InferenceParams, InferenceSettings
from ck_engine.client import EngineClient


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <project_path>")
        sys.exit(1)

    project_path = sys.argv[1]

    # Spawn the engine as a subprocess — zero configuration.
    # The engine process communicates over stdin/stdout using
    # JSON-RPC 2.0 with Content-Length framing.
    with EngineClient.spawn() as engine:
        # 1. Query engine capabilities
        caps = engine.capabilities()
        print(f"Engine v{caps['version']}")
        print(f"  Generators: {', '.join(caps['generators'])}")
        print(f"  Backends:   {', '.join(caps['backends'])}")
        print()

        # 2. Scan project for clips
        project = engine.scan_project(project_path)
        clips = project["clips"]
        print(f"Found {len(clips)} clip(s) in {project['project_path']}:")
        for clip in clips:
            inp = clip.get("input")
            alpha = clip.get("alpha")
            inp_frames = inp["frame_count"] if inp else 0
            alpha_frames = alpha["frame_count"] if alpha else 0
            print(f"  {clip['name']}: {inp_frames} input, {alpha_frames} alpha frames")
        print()

        # 3. Generate alpha mattes (if needed)
        clips_needing_alpha = [c for c in clips if c.get("input") and not c.get("alpha")]
        if clips_needing_alpha:
            print("Generating alpha mattes with BiRefNet...")
            job_id = engine.submit_generate(GenerateParams(
                path=project_path,
                model="birefnet",
                mode="replace",
            ))
            _wait_for_job(engine)
            print()

        # 4. Run inference
        print("Running inference...")
        job_id = engine.submit_inference(InferenceParams(
            path=project_path,
            settings=InferenceSettings(
                input_is_linear=False,
                despill_strength=0.5,
                auto_despeckle=True,
                refiner_scale=1.0,
            ),
        ))
        _wait_for_job(engine)

        # 5. Check model status
        models = engine.model_status()
        if models.get("inference_engine"):
            eng = models["inference_engine"]
            print(f"\nModel loaded: {eng['backend']} on {eng['device']} ({eng['vram_mb']:.0f} MB)")


def _wait_for_job(engine: EngineClient):
    """Block until the active job completes, printing progress."""
    import queue as _queue

    try:
        for event in engine.iter_events(timeout=600.0):
            if isinstance(event, ModelLoading):
                print(f"  Loading {event.model} on {event.device}...")
            elif isinstance(event, ModelLoaded):
                print(f"  Model loaded in {event.load_seconds:.1f}s ({event.vram_mb:.0f} MB)")
            elif isinstance(event, ClipStarted):
                print(f"  Processing: {event.clip} ({event.frames} frames)")
            elif isinstance(event, JobProgress):
                pct = event.done / event.total * 100 if event.total > 0 else 0
                fps_str = f" @ {event.fps:.1f} fps" if event.fps > 0 else ""
                print(f"\r  [{event.done}/{event.total}] {pct:.0f}%{fps_str}", end="", flush=True)
            elif isinstance(event, LogEvent):
                if event.level in ("warning", "error"):
                    print(f"\n  [{event.level.upper()}] {event.message}")
            elif isinstance(event, JobCompleted):
                print(f"\n  Done: {event.clips_ok} clip(s), {event.frames_ok} frames in {event.elapsed_seconds:.1f}s")
                return
            elif isinstance(event, JobFailed):
                print(f"\n  FAILED: {event.error}")
                return
            elif isinstance(event, JobCancelled):
                print(f"\n  Cancelled after {event.frames_completed} frames")
                return
    except _queue.Empty:
        print("\n  Timed out waiting for engine")


if __name__ == "__main__":
    main()
