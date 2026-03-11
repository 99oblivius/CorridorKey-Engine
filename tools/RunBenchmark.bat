@echo off
REM 4K Benchmark: Flash Attention baseline vs All Optimizations
REM Requires: Tears of Steel test frames + alpha hints
REM Run tools\benchmarks\download_frames.py and tools\benchmarks\generate_alpha_hints.py first

cd /d "%~dp0\.."
uv run python tools\benchmarks\benchmark_4k_vram.py
pause
