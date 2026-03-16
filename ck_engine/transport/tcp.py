"""TCP transport — reads/writes JSON-RPC over a TCP socket.

Uses the same Content-Length framing as the stdio transport.  This is
the transport for daemon mode (``corridorkey-engine --listen :9400``).

Supports both server (listen/accept) and client (connect) modes.
"""

from __future__ import annotations

import json
import socket
import threading

from . import Transport, TransportClosed, TransportError


def _parse_address(address: str) -> tuple[str, int]:
    """Parse 'host:port' or ':port' into (host, port)."""
    if address.startswith(":"):
        return ("0.0.0.0", int(address[1:]))
    host, _, port_str = address.rpartition(":")
    if not port_str:
        raise ValueError(f"invalid address: {address!r} (expected host:port)")
    return (host or "0.0.0.0", int(port_str))


class TcpTransport(Transport):
    """JSON-RPC transport over a TCP socket using Content-Length framing."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._rfile = sock.makefile("rb")
        self._write_lock = threading.Lock()
        self._closed = False
        self._server_sock: socket.socket | None = None

    @classmethod
    def listen(cls, address: str) -> TcpTransport:
        """Bind and accept one connection. Blocks until a client connects.

        Address format: "host:port" or ":port" (bind all interfaces).
        """
        host, port = _parse_address(address)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        conn, _addr = server.accept()
        transport = cls(conn)
        transport._server_sock = server
        return transport

    @classmethod
    def connect(cls, address: str) -> TcpTransport:
        """Connect to a listening engine.

        Address format: "host:port".
        """
        host, port = _parse_address(address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
        except OSError as exc:
            sock.close()
            raise TransportError(f"cannot connect to {host}:{port}: {exc}") from exc
        return cls(sock)

    def read_message(self) -> dict | None:
        if self._closed:
            raise TransportClosed("transport is closed")

        content_length = -1
        while True:
            line = self._read_line()
            if line is None:
                return None  # connection closed
            if line == b"":
                break  # end of headers
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except (ValueError, IndexError) as exc:
                    raise TransportError(f"invalid Content-Length: {line!r}") from exc

        if content_length < 0:
            raise TransportError("missing Content-Length header")

        body = self._read_exact(content_length)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise TransportError(f"invalid JSON: {exc}") from exc

    def _read_line(self) -> bytes | None:
        """Read one \\r\\n terminated line. Returns None on EOF."""
        buf = bytearray()
        while True:
            b = self._rfile.read(1)
            if not b:
                return None if not buf else bytes(buf)
            buf.extend(b)
            if buf.endswith(b"\r\n"):
                return bytes(buf[:-2])

    def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._rfile.read(n - len(buf))
            if not chunk:
                raise TransportError(f"unexpected EOF: wanted {n}, got {len(buf)}")
            buf.extend(chunk)
        return bytes(buf)

    def write_message(self, msg: dict) -> None:
        if self._closed:
            raise TransportClosed("transport is closed")
        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            try:
                self._sock.sendall(header + body)
            except OSError as exc:
                raise TransportError(f"write failed: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Shut down the socket for reading first — this unblocks any
        # thread stuck in _rfile.read() immediately, rather than waiting
        # for the peer to close its end.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._rfile.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

    @property
    def is_open(self) -> bool:
        return not self._closed
