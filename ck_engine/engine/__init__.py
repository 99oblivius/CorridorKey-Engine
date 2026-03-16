"""CorridorKey engine process.

The engine is a long-running process that accepts JSON-RPC commands
over stdio or TCP and executes CorridorKey pipeline operations.
"""

from ck_engine.engine.server import EngineServer, main

__all__ = ["EngineServer", "main"]
