"""JSON-RPC method routing for the CorridorKey engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ck_engine.api.errors import (
    CANCELLED,
    ENGINE_BUSY,
    INVALID_PARAMS,
    INVALID_REQUEST,
    JOB_NOT_FOUND,
    METHOD_NOT_FOUND,
    EngineError,
    error_response,
    success_response,
)

if TYPE_CHECKING:
    from ck_engine.engine.server import EngineServer

logger = logging.getLogger(__name__)


class Dispatcher:
    """Routes JSON-RPC requests to handler methods."""

    def __init__(self, server: EngineServer) -> None:
        self._server = server
        self._handlers: dict[str, Any] = {
            "engine.capabilities": self._handle_capabilities,
            "engine.status": self._handle_status,
            "engine.shutdown": self._handle_shutdown,
            "job.submit": self._handle_job_submit,
            "job.cancel": self._handle_job_cancel,
            "job.status": self._handle_job_status,
            "model.status": self._handle_model_status,
            "model.unload": self._handle_model_unload,
            "events.subscribe": self._handle_events_subscribe,
            "events.unsubscribe": self._handle_events_unsubscribe,
            "project.scan": self._handle_project_scan,
        }

    def dispatch(self, message: dict) -> dict | None:
        """Dispatch a JSON-RPC message. Returns response dict or None for notifications."""
        # Validate basic JSON-RPC structure
        if not isinstance(message, dict):
            return error_response(None, INVALID_REQUEST, "Request must be a JSON object")

        if message.get("jsonrpc") != "2.0":
            return error_response(
                message.get("id"), INVALID_REQUEST, "Missing or invalid jsonrpc version"
            )

        method = message.get("method")
        if not isinstance(method, str):
            return error_response(
                message.get("id"), INVALID_REQUEST, "Missing or invalid method"
            )

        request_id = message.get("id")
        params = message.get("params", {})

        # Notifications (no id) — we don't handle any client notifications currently
        if request_id is None:
            return None

        # Find handler
        handler = self._handlers.get(method)
        if handler is None:
            return error_response(request_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

        # Execute handler
        try:
            result = handler(params, request_id)
            return success_response(request_id, result)
        except EngineError as exc:
            return exc.to_response(request_id)
        except Exception as exc:
            logger.exception("Unhandled error in %s", method)
            return error_response(request_id, -32603, f"Internal error: {exc}")

    def _handle_capabilities(self, params: dict, request_id: Any) -> dict:
        """Handle engine.capabilities request."""
        from ck_engine.device import detect_best_device

        # Detect available devices
        devices = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "id": f"cuda:{i}",
                        "name": props.name,
                        "vram_gb": round(props.total_mem / (1024**3), 1),
                    })
        except Exception:
            pass

        if not devices:
            best = detect_best_device()
            devices.append({"id": best, "name": best, "vram_gb": 0.0})

        return {
            "version": "2.0.0",
            "generators": ["birefnet", "gvm", "videomama"],
            "backends": ["torch", "torch_optimized", "mlx"],
            "devices": devices,
            "profiles": ["original", "optimized", "performance", "experimental"],
            "transport": "stdio",  # TODO: detect from server
        }

    def _handle_status(self, params: dict, request_id: Any) -> dict:
        """Handle engine.status request."""
        return {
            "state": self._server.state,
            "active_job": self._server.active_job_id,
            "models_loaded": self._server.model_pool.status(),
            "vram": None,  # TODO: populate from model_pool
            "uptime_seconds": round(self._server.uptime, 2),
        }

    def _handle_shutdown(self, params: dict, request_id: Any) -> str:
        """Handle engine.shutdown request."""
        self._server.request_shutdown()
        return "ok"

    def _handle_job_submit(self, params: dict, request_id: Any) -> dict:
        """Handle job.submit request."""
        job_type = params.get("type")
        if job_type not in ("generate", "inference"):
            raise EngineError(INVALID_PARAMS, f"Invalid job type: {job_type}")

        if self._server.state != "idle":
            raise EngineError(
                ENGINE_BUSY, "Engine busy",
                {"active_job": self._server.active_job_id},
            )

        if job_type == "generate":
            job_params = _parse_generate_params(params)
        else:
            job_params = _parse_inference_params(params)

        # Scan clips to return in response
        clip_info = self._server.job_runner.scan_project(params.get("path", ""))

        # Start the job
        job_id = self._server.start_job(job_type, job_params)

        return {
            "job_id": job_id,
            "clips": clip_info["clips"],
            "total_frames": sum(
                c.get("input", {}).get("frame_count", 0) if c.get("input") else 0
                for c in clip_info["clips"]
            ),
        }

    def _handle_job_cancel(self, params: dict, request_id: Any) -> str:
        """Handle job.cancel request."""
        job_id = params.get("job_id", "")
        if not self._server.active_job_id or self._server.active_job_id != job_id:
            raise EngineError(JOB_NOT_FOUND, f"No active job with id: {job_id}")
        self._server.cancel_job()
        return "cancelling"

    def _handle_job_status(self, params: dict, request_id: Any) -> dict:
        """Handle job.status request."""
        job_id = params.get("job_id", "")
        status = self._server.get_job_status(job_id)
        if status is None:
            raise EngineError(JOB_NOT_FOUND, f"No job with id: {job_id}")
        return status

    def _handle_model_status(self, params: dict, request_id: Any) -> dict:
        """Handle model.status request."""
        return self._server.model_pool.status()

    def _handle_model_unload(self, params: dict, request_id: Any) -> dict:
        """Handle model.unload request."""
        which = params.get("which", "all")
        if which not in ("all", "inference", "generator"):
            raise EngineError(INVALID_PARAMS, f"Invalid unload target: {which}")
        freed = self._server.model_pool.unload(which)
        return {"freed_mb": round(freed, 1)}

    def _handle_events_subscribe(self, params: dict, request_id: Any) -> str:
        """Handle events.subscribe request."""
        categories = params.get("categories", [])
        self._server.event_bus.subscribe(categories)
        return "ok"

    def _handle_events_unsubscribe(self, params: dict, request_id: Any) -> str:
        """Handle events.unsubscribe request."""
        categories = params.get("categories", [])
        self._server.event_bus.unsubscribe(categories)
        return "ok"

    def _handle_project_scan(self, params: dict, request_id: Any) -> dict:
        """Handle project.scan request."""
        path = params.get("path", "")
        return self._server.job_runner.scan_project(path)


def _parse_generate_params(raw: dict) -> "GenerateParams":
    """Parse raw dict into GenerateParams."""
    from ck_engine.api.types import GenerateParams
    return GenerateParams(
        path=raw.get("path", ""),
        model=raw.get("model", "birefnet"),
        mode=raw.get("mode", "replace"),
        frames=raw.get("frames"),
        device=raw.get("device", "auto"),
        halt_on_failure=raw.get("halt_on_failure", False),
    )


def _parse_inference_params(raw: dict) -> "InferenceParams":
    """Parse raw dict into InferenceParams."""
    from ck_engine.api.types import InferenceParams, InferenceSettings, OptimizationParams

    settings_raw = raw.get("settings", {})
    settings = InferenceSettings(
        input_is_linear=settings_raw.get("input_is_linear", False),
        despill_strength=settings_raw.get("despill_strength", 0.5),
        auto_despeckle=settings_raw.get("auto_despeckle", True),
        despeckle_size=settings_raw.get("despeckle_size", 400),
        refiner_scale=settings_raw.get("refiner_scale", 1.0),
    )

    opt_raw = raw.get("optimization")
    optimization = None
    if opt_raw:
        optimization = OptimizationParams.from_dict(opt_raw)

    return InferenceParams(
        path=raw.get("path", ""),
        frames=raw.get("frames"),
        device=raw.get("device", "auto"),
        backend=raw.get("backend", "auto"),
        settings=settings,
        optimization=optimization,
        devices=raw.get("devices"),
        img_size=raw.get("img_size", 2048),
        read_workers=raw.get("read_workers", 0),
        write_workers=raw.get("write_workers", 0),
        cpus=raw.get("cpus", 0),
        gpu_resilience=raw.get("gpu_resilience", False),
        halt_on_failure=raw.get("halt_on_failure", False),
    )
