"""Tests for backend.transport — StdioTransport and TcpTransport."""

from __future__ import annotations

import io
import json
import socket
import threading
import time

import pytest

from ck_engine.transport import TransportClosed, TransportError
from ck_engine.transport.stdio import StdioTransport
from ck_engine.transport.tcp import TcpTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(msg: dict) -> bytes:
    """Build a Content-Length framed message."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _get_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# TestStdioTransport
# ---------------------------------------------------------------------------


class TestStdioTransport:
    def test_write_then_read_round_trip(self):
        """Write a message to BytesIO stdout, read it back as stdin."""
        out = io.BytesIO()
        writer = StdioTransport(io.BytesIO(), out)
        msg = {"jsonrpc": "2.0", "method": "ping", "id": 1}
        writer.write_message(msg)

        out.seek(0)
        reader = StdioTransport(out, io.BytesIO())
        received = reader.read_message()
        assert received == msg

    def test_multiple_messages(self):
        """Write 3 messages, read them back in order."""
        out = io.BytesIO()
        writer = StdioTransport(io.BytesIO(), out)
        messages = [
            {"jsonrpc": "2.0", "method": "a", "id": 1},
            {"jsonrpc": "2.0", "method": "b", "id": 2},
            {"jsonrpc": "2.0", "result": "done", "id": 3},
        ]
        for m in messages:
            writer.write_message(m)

        out.seek(0)
        reader = StdioTransport(out, io.BytesIO())
        for expected in messages:
            assert reader.read_message() == expected

    def test_large_message(self):
        """Message with a 100 KB string value round-trips correctly."""
        out = io.BytesIO()
        writer = StdioTransport(io.BytesIO(), out)
        msg = {"jsonrpc": "2.0", "method": "big", "data": "x" * 100_000, "id": 99}
        writer.write_message(msg)

        out.seek(0)
        reader = StdioTransport(out, io.BytesIO())
        received = reader.read_message()
        assert received == msg

    def test_eof_returns_none(self):
        """Empty stdin causes read_message() to return None."""
        transport = StdioTransport(io.BytesIO(b""), io.BytesIO())
        assert transport.read_message() is None

    def test_malformed_header(self):
        """stdin with no Content-Length header raises TransportError."""
        # The header line is not recognized; blank line follows, then EOF.
        # read_message() will reach the blank line, find content_length == -1,
        # and raise TransportError("missing Content-Length header").
        bad = b"Bad-Header: foo\r\n\r\n{}"
        transport = StdioTransport(io.BytesIO(bad), io.BytesIO())
        with pytest.raises(TransportError):
            transport.read_message()

    def test_invalid_json(self):
        """Valid Content-Length header but non-JSON body raises TransportError."""
        body = b"not valid json!!!"
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        transport = StdioTransport(io.BytesIO(header + body), io.BytesIO())
        with pytest.raises(TransportError):
            transport.read_message()

    def test_closed_read_raises(self):
        """read_message() after close() raises TransportClosed."""
        transport = StdioTransport(io.BytesIO(), io.BytesIO())
        transport.close()
        with pytest.raises(TransportClosed):
            transport.read_message()

    def test_closed_write_raises(self):
        """write_message() after close() raises TransportClosed."""
        transport = StdioTransport(io.BytesIO(), io.BytesIO())
        transport.close()
        with pytest.raises(TransportClosed):
            transport.write_message({"id": 1})

    def test_is_open(self):
        """is_open is True after init and False after close."""
        transport = StdioTransport(io.BytesIO(), io.BytesIO())
        assert transport.is_open is True
        transport.close()
        assert transport.is_open is False

    def test_write_produces_content_length_framing(self):
        """Written bytes contain a proper Content-Length header."""
        out = io.BytesIO()
        transport = StdioTransport(io.BytesIO(), out)
        msg = {"jsonrpc": "2.0", "method": "test", "id": 42}
        transport.write_message(msg)

        raw = out.getvalue()
        # Must start with "Content-Length: "
        assert raw.startswith(b"Content-Length: ")
        # Header and body are separated by \r\n\r\n
        assert b"\r\n\r\n" in raw
        header_part, body_part = raw.split(b"\r\n\r\n", 1)
        declared_length = int(header_part.split(b":", 1)[1].strip())
        assert declared_length == len(body_part)
        assert json.loads(body_part) == msg


# ---------------------------------------------------------------------------
# TestTcpTransport
# ---------------------------------------------------------------------------


class TestTcpTransport:
    def _start_server(self, address: str, server_holder: list) -> threading.Thread:
        """Spawn a thread that calls TcpTransport.listen(address)."""

        def _listen():
            server_holder[0] = TcpTransport.listen(address)

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        return t

    def test_listen_connect_round_trip(self):
        """Bidirectional messaging between server and client."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]
        t = self._start_server(f"127.0.0.1:{port}", server)

        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        msg = {"jsonrpc": "2.0", "method": "test", "id": 1}
        client.write_message(msg)
        received = server[0].read_message()
        assert received == msg

        reply = {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        server[0].write_message(reply)
        received = client.read_message()
        assert received == reply

        client.close()
        server[0].close()

    def test_multiple_messages_tcp(self):
        """Send 5 messages from client to server, receive all 5."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]
        t = self._start_server(f"127.0.0.1:{port}", server)

        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        messages = [{"id": i, "method": f"op{i}"} for i in range(5)]
        for m in messages:
            client.write_message(m)

        received = [server[0].read_message() for _ in range(5)]
        assert received == messages

        client.close()
        server[0].close()

    def test_large_message_tcp(self):
        """100 KB message round-trips over TCP."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]
        t = self._start_server(f"127.0.0.1:{port}", server)

        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        msg = {"jsonrpc": "2.0", "method": "big", "data": "y" * 100_000, "id": 7}
        client.write_message(msg)
        received = server[0].read_message()
        assert received == msg

        client.close()
        server[0].close()

    def test_client_close_eof(self):
        """After client closes, server read_message() returns None."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]
        t = self._start_server(f"127.0.0.1:{port}", server)

        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        client.close()
        # Server should get EOF (None), not an exception.
        result = server[0].read_message()
        assert result is None

        server[0].close()

    def test_is_open_tcp(self):
        """is_open is True after connect and False after close."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]
        t = self._start_server(f"127.0.0.1:{port}", server)

        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        assert client.is_open is True
        client.close()
        assert client.is_open is False

        server[0].close()

    def test_address_parsing(self):
        """_parse_address handles ':port' and 'host:port' forms."""
        from ck_engine.transport.tcp import _parse_address

        assert _parse_address(":9400") == ("0.0.0.0", 9400)
        assert _parse_address("localhost:9400") == ("localhost", 9400)
        assert _parse_address("127.0.0.1:8080") == ("127.0.0.1", 8080)


# ---------------------------------------------------------------------------
# TestConcurrentWrites
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    def test_concurrent_writes_stdio(self):
        """Multiple threads writing simultaneously don't corrupt framing."""
        out = io.BytesIO()
        transport = StdioTransport(io.BytesIO(), out)

        errors: list[Exception] = []

        def _write(i: int) -> None:
            try:
                transport.write_message({"id": i, "data": "x" * 1000})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        # Read all messages back — should be 20 valid messages
        out.seek(0)
        reader = StdioTransport(out, io.BytesIO())
        messages = []
        while True:
            msg = reader.read_message()
            if msg is None:
                break
            messages.append(msg)

        assert len(messages) == 20
        assert sorted(m["id"] for m in messages) == list(range(20))

    def test_concurrent_writes_tcp(self):
        """Multiple threads writing to a TcpTransport don't corrupt framing."""
        port = _get_free_port()
        server: list[TcpTransport | None] = [None]

        def _listen():
            server[0] = TcpTransport.listen(f"127.0.0.1:{port}")

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        time.sleep(0.1)
        client = TcpTransport.connect(f"127.0.0.1:{port}")
        t.join(timeout=2)

        errors: list[Exception] = []

        def _write(i: int) -> None:
            try:
                client.write_message({"id": i, "data": "z" * 1000})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        messages = []
        for _ in range(20):
            msg = server[0].read_message()
            assert msg is not None
            messages.append(msg)

        assert len(messages) == 20
        assert sorted(m["id"] for m in messages) == list(range(20))

        client.close()
        server[0].close()
