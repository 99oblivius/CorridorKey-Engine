"""Public API for the CorridorKey engine protocol.

This package defines the contract between the engine process and its
frontends.  Import types, events, and helpers from here::

    from ck_engine.api import GenerateParams, InferenceParams, JobCompleted
"""

from __future__ import annotations

# --- Types ---
from .types import (
    AssetInfo,
    ClipInfo,
    DeviceInfo,
    EngineCapabilities,
    EngineStatus,
    FailedFrameInfo,
    GenerateParams,
    InferenceParams,
    InferenceSettings,
    JobResult,
    JobStatus,
    LoadedModelInfo,
    OptimizationParams,
    VRAMInfo,
)

# --- Events ---
from .events import (
    ALL_EVENT_TYPES,
    EVENT_CATEGORIES,
    ClipCompleted,
    ClipStarted,
    Event,
    JobAccepted,
    JobCancelled,
    JobCompleted,
    JobFailed,
    JobProgress,
    LogEvent,
    ModelLoaded,
    ModelLoading,
    ModelRecompiling,
    ModelUnloaded,
    WarningEvent,
)

# --- Errors ---
from .errors import (
    CANCELLED,
    DEVICE_UNAVAILABLE,
    ENGINE_BUSY,
    INVALID_PARAMS,
    INVALID_PATH,
    INVALID_REQUEST,
    JOB_NOT_FOUND,
    METHOD_NOT_FOUND,
    MODEL_LOAD_FAILURE,
    NO_VALID_CLIPS,
    PARSE_ERROR,
    EngineError,
    error_response,
    success_response,
)

# --- Frame range utilities ---
from .frames import format_frame_range, parse_frame_range

__all__ = [
    # Types
    "AssetInfo",
    "ClipInfo",
    "DeviceInfo",
    "EngineCapabilities",
    "EngineStatus",
    "FailedFrameInfo",
    "GenerateParams",
    "InferenceParams",
    "InferenceSettings",
    "JobResult",
    "JobStatus",
    "LoadedModelInfo",
    "OptimizationParams",
    "VRAMInfo",
    # Events
    "ALL_EVENT_TYPES",
    "EVENT_CATEGORIES",
    "ClipCompleted",
    "ClipStarted",
    "Event",
    "JobAccepted",
    "JobCancelled",
    "JobCompleted",
    "JobFailed",
    "JobProgress",
    "LogEvent",
    "ModelLoaded",
    "ModelLoading",
    "ModelRecompiling",
    "ModelUnloaded",
    "WarningEvent",
    # Errors
    "CANCELLED",
    "DEVICE_UNAVAILABLE",
    "ENGINE_BUSY",
    "INVALID_PARAMS",
    "INVALID_PATH",
    "INVALID_REQUEST",
    "JOB_NOT_FOUND",
    "METHOD_NOT_FOUND",
    "MODEL_LOAD_FAILURE",
    "NO_VALID_CLIPS",
    "PARSE_ERROR",
    "EngineError",
    "error_response",
    "success_response",
    # Frames
    "format_frame_range",
    "parse_frame_range",
]
