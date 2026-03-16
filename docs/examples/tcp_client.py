#!/usr/bin/env python3
"""CorridorKey Engine — TCP client example.

Connects to a running engine daemon over TCP.
Start the daemon first:  corridorkey-engine serve --listen :9400

Prerequisites:
    pip install corridorkey-engine
    corridorkey-engine serve --listen :9400   # in another terminal

Usage:
    python docs/examples/tcp_client.py localhost:9400 /path/to/project
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
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <host:port> <project_path>")
        print(f"Example: {sys.argv[0]} localhost:9400 /path/to/clips")
        sys.exit(1)

    address = sys.argv[1]
    project_path = sys.argv[2]

    # Connect to a running engine daemon.
    # The daemon must be started separately:
    #   corridorkey-engine serve --listen :9400
    print(f"Connecting to {address}...")
    engine = EngineClient.connect(address)

    try:
        # Query capabilities
        caps = engine.capabilities()
        print(f"Connected to engine v{caps['version']}")
        print(f"  Transport: {caps['transport']}")
        print()

        # Scan project
        project = engine.scan_project(project_path)
        clips = project["clips"]
        print(f"Found {len(clips)} clip(s):")
        for clip in clips:
            inp = clip.get("input")
            frames = inp["frame_count"] if inp else 0
            print(f"  {clip['name']}: {frames} frames")
        print()

        # Submit inference job
        print("Submitting inference job...")
        job_id = engine.submit_inference(
            InferenceParams(
                path=project_path,
                settings=InferenceSettings(despill_strength=0.5),
            )
        )
        print(f"Job accepted: {job_id}")

        # Stream events
        import queue as _queue

        try:
            for event in engine.iter_events(timeout=600.0):
                if isinstance(event, ClipStarted):
                    print(f"  Clip: {event.clip} ({event.frames} frames)")
                elif isinstance(event, JobProgress):
                    pct = event.done / event.total * 100 if event.total > 0 else 0
                    fps_str = f" @ {event.fps:.1f} fps" if event.fps > 0 else ""
                    print(f"\r  [{event.done}/{event.total}] {pct:.0f}%{fps_str}", end="", flush=True)
                elif isinstance(event, JobCompleted):
                    print(f"\n  Completed in {event.elapsed_seconds:.1f}s")
                    break
                elif isinstance(event, JobFailed):
                    print(f"\n  Failed: {event.error}")
                    break
                elif isinstance(event, JobCancelled):
                    print(f"\n  Cancelled")
                    break
                elif isinstance(event, LogEvent):
                    if event.level in ("warning", "error"):
                        print(f"\n  [{event.level.upper()}] {event.message}")
        except _queue.Empty:
            print("\n  Timed out")

        # Check engine status
        status = engine.status()
        print(f"\nEngine state: {status['state']}")

    finally:
        # Don't shutdown — the daemon stays running for other clients.
        engine.close(shutdown=False)
        print("Disconnected.")


if __name__ == "__main__":
    main()
