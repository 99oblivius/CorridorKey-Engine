#!/usr/bin/env bash
# CorridorKey — quick launcher.
# Forwards all arguments to tools/cli.sh.
# Usage: ./launch.sh [subcommand] [options]
#   ./launch.sh                          # TUI
#   ./launch.sh inference /path/to/clips  # headless keying
#   ./launch.sh --help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/tools/cli.sh" "$@"
