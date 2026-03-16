"""Transport layer for the CorridorKey engine protocol.

A transport moves JSON-RPC messages between engine and client over some
byte channel (stdio pipes, TCP socket, etc.).  All transports use the
same Content-Length framing so the upper layers never think about bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TransportClosed(Exception):
    """Raised when a read/write is attempted on a closed transport."""


class TransportError(Exception):
    """Raised on framing or I/O errors."""


class Transport(ABC):
    """Abstract base for JSON-RPC message transports."""

    @abstractmethod
    def read_message(self) -> dict | None:
        """Read one JSON-RPC message.

        Returns the parsed dict, or ``None`` if the peer closed the
        connection cleanly (EOF).

        Raises
        ------
        TransportClosed
            If the transport has been closed locally.
        TransportError
            On framing or deserialization errors.
        """

    @abstractmethod
    def write_message(self, msg: dict) -> None:
        """Write one JSON-RPC message.

        Raises
        ------
        TransportClosed
            If the transport has been closed.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the transport.  Idempotent."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the transport is still open for reads/writes."""
