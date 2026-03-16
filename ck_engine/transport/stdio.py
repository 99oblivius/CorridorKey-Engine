"""Stdio transport — reads/writes JSON-RPC over stdin/stdout.

Uses Content-Length framing identical to the Language Server Protocol::

    Content-Length: 42\r\n
    \r\n
    {"jsonrpc":"2.0","method":"engine.status","id":1}

This is the default transport for subprocess mode (TUI spawns engine).
"""

from __future__ import annotations

import json
import threading
from typing import BinaryIO

from . import Transport, TransportClosed, TransportError


class StdioTransport(Transport):
    """JSON-RPC transport over a pair of binary stdio streams."""

    def __init__(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        self._in = stdin
        self._out = stdout
        self._write_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_line(self) -> bytes | None:
        """Read one line ending in \\r\\n.

        Returns the line *without* the trailing ``\\r\\n``, or ``None`` on
        EOF.  Raises :exc:`TransportError` on a partial read (EOF mid-line).
        """
        buf = bytearray()
        while True:
            b = self._in.read(1)
            if not b:
                # EOF
                if not buf:
                    return None
                raise TransportError("unexpected EOF mid-header")
            buf.extend(b)
            if buf.endswith(b"\r\n"):
                return bytes(buf[:-2])

    def _read_exact(self, n: int) -> bytes:
        """Read exactly *n* bytes, handling partial reads."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._in.read(n - len(buf))
            if not chunk:
                raise TransportError(
                    f"unexpected EOF: expected {n} bytes, got {len(buf)}"
                )
            buf.extend(chunk)
        return bytes(buf)

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

    def read_message(self) -> dict | None:
        """Read one LSP-framed JSON-RPC message from stdin.

        Returns the parsed dict, or ``None`` on clean EOF.

        Raises
        ------
        TransportClosed
            If the transport has already been closed locally.
        TransportError
            On framing or JSON deserialization errors.
        """
        if self._closed:
            raise TransportClosed("transport is closed")

        # Read headers until the blank separator line.
        content_length = -1
        while True:
            line = self._read_line()
            if line is None:
                return None  # clean EOF
            if line == b"":
                break  # blank line — end of headers
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except (ValueError, IndexError) as exc:
                    raise TransportError(
                        f"invalid Content-Length header: {line!r}"
                    ) from exc

        if content_length < 0:
            raise TransportError("missing Content-Length header")

        body = self._read_exact(content_length)

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise TransportError(f"invalid JSON: {exc}") from exc

    def write_message(self, msg: dict) -> None:
        """Write one JSON-RPC message to stdout with Content-Length framing.

        Raises
        ------
        TransportClosed
            If the transport has been closed.
        """
        if self._closed:
            raise TransportClosed("transport is closed")
        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            self._out.write(header + body)
            self._out.flush()

    def close(self) -> None:
        """Close both streams.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._in.close()
        except Exception:
            pass
        try:
            self._out.close()
        except Exception:
            pass

    @property
    def is_open(self) -> bool:
        """Whether the transport is still open for reads/writes."""
        return not self._closed
