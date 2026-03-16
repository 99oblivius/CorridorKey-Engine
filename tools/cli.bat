@echo off
REM CorridorKey CLI launcher — runs the Typer TUI via uv.
REM
REM Called by launch.bat (root) or directly from tools\.
REM Append any corridorkey subcommand and flags after the script name.
REM
REM Default (wizard):
REM   tools\cli.bat C:\path\to\clips
REM   tools\cli.bat --profile optimized C:\path\to\clips
REM
REM Subcommands:
REM   tools\cli.bat inference --linear --despill 8
REM   tools\cli.bat generate-alphas --model gvm
REM   tools\cli.bat generate-alphas --model birefnet
REM   tools\cli.bat generate-alphas --model videomama
REM   tools\cli.bat list
REM   tools\cli.bat --help
REM
REM Global options (before subcommand):
REM   --device TEXT       Compute device: auto, cuda, mps, cpu  [default: auto]
REM   --backend TEXT      Backend: auto, torch, torch_optimized, mlx
REM   --devices TEXT      Comma-separated GPU indices (e.g. 0,1)
REM   --profile TEXT      Optimization profile: optimized, original, ...
REM   --img-size INT      Model input resolution [default: 2048]
REM   --read-workers INT  Reader thread pool size (0=auto)
REM   --write-workers INT Writer thread pool size (0=auto)
REM   (+ optimization flags — see corridorkey --help)
REM
REM Environment:
REM   Requires uv (https://docs.astral.sh/uv/).
REM   Python, venv, and dependencies are managed automatically by uv.

REM Enable OpenEXR support
set "OPENCV_IO_ENABLE_OPENEXR=1"

REM Resolve project root (one level up from tools\)
cd /d "%~dp0\.."

uv run corridorkey-engine %*
