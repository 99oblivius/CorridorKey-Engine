#!/usr/bin/env bash
# CorridorKey CLI launcher — runs the Typer TUI via uv.
#
# Called by launch.sh (root) or directly from tools/.
# Append any corridorkey subcommand and flags after the script name.
#
# Default (wizard):
#   ./tools/cli.sh /path/to/clips
#   ./tools/cli.sh --profile optimized /path/to/clips
#
# Subcommands:
#   ./tools/cli.sh inference --linear --despill 8
#   ./tools/cli.sh generate-alphas --model gvm
#   ./tools/cli.sh generate-alphas --model birefnet
#   ./tools/cli.sh generate-alphas --model videomama
#   ./tools/cli.sh list
#   ./tools/cli.sh --help
#
# Global options (before subcommand):
#   --device TEXT       Compute device: auto, cuda, mps, cpu  [default: auto]
#   --backend TEXT      Backend: auto, torch, torch_optimized, mlx
#   --devices TEXT      Comma-separated GPU indices (e.g. 0,1)
#   --profile TEXT      Optimization profile: optimized, original, ...
#   --img-size INT      Model input resolution [default: 2048]
#   --read-workers INT  Reader thread pool size (0=auto)
#   --write-workers INT Writer thread pool size (0=auto)
#   (+ optimization flags — see corridorkey --help)
#
# Environment:
#   Requires uv (https://docs.astral.sh/uv/).
#   Python, venv, and dependencies are managed automatically by uv.

set -e

# Resolve project root (one level up from tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Enable OpenEXR support
export OPENCV_IO_ENABLE_OPENEXR=1

cd "$PROJECT_ROOT"
exec uv run corridorkey-engine "$@"
