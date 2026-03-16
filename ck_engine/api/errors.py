"""JSON-RPC error codes and helpers for the CorridorKey engine protocol."""

from __future__ import annotations

# --- Standard JSON-RPC errors ---
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

# --- Application errors ---
ENGINE_BUSY = -32000
JOB_NOT_FOUND = -32001
INVALID_PATH = -32002
NO_VALID_CLIPS = -32003
MODEL_LOAD_FAILURE = -32004
DEVICE_UNAVAILABLE = -32005
CANCELLED = -32006


class EngineError(Exception):
    """Error with a JSON-RPC error code attached."""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_response(self, request_id: int | str | None) -> dict:
        """Build a JSON-RPC error response dict."""
        err: dict = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return {"jsonrpc": "2.0", "id": request_id, "error": err}


def error_response(
    request_id: int | str | None,
    code: int,
    message: str,
    data: object = None,
) -> dict:
    """Build a JSON-RPC error response dict."""
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": err}


def success_response(request_id: int | str | None, result: object) -> dict:
    """Build a JSON-RPC success response dict."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
