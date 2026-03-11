#!/usr/bin/env bash

# 4K Benchmark: Flash Attention baseline vs All Optimizations
# Requires: Tears of Steel test frames + alpha hints
# Run tools/benchmarks/download_frames.py and tools/benchmarks/generate_alpha_hints.py first

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

uv run python tools/benchmarks/benchmark_4k_vram.py
