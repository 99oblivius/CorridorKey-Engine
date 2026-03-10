"""Tests for async pipeline data transfer contracts.

These tests verify the invariants that broke in practice during development:
  - PendingTransfer must release buffer slots and GPU bulk references
  - Drain workers must shut down cleanly via sentinel cascade

No GPU or model weights required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np

from CorridorKeyModule.base_engine import PendingTransfer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(h: int = 4, w: int = 4) -> dict:
    """Minimal valid result dict."""
    return {
        "alpha": np.full((h, w, 1), 0.8, dtype=np.float32),
        "fg": np.full((h, w, 3), 0.6, dtype=np.float32),
        "comp": np.full((h, w, 3), 0.5, dtype=np.float32),
        "processed": np.full((h, w, 4), 0.4, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# PendingTransfer lifecycle
# ---------------------------------------------------------------------------


class TestPendingTransfer:
    """Verify the PendingTransfer buffer lifecycle contract.

    These invariants were each discovered via frame corruption bugs:
      - _buf_released must be set after resolve() so the engine can reuse
        the pinned buffer slot without overwriting in-flight data.
      - _gpu_bulk must be cleared after DMA sync so the CUDA caching
        allocator can reclaim the GPU memory, but NOT before (holding it
        prevents the allocator from reusing memory the DMA is still reading).
    """

    def test_cpu_fallback_resolves_immediately(self):
        """CPU result (no DMA) resolves without blocking."""
        result = _fake_result()
        pt = PendingTransfer(_event=None, _pinned_buf=None, _cpu_result=result, _buf_released=None)
        resolved = pt.resolve()
        assert resolved is result
        assert resolved["alpha"].shape == (4, 4, 1)

    def test_resolve_returns_same_result_on_repeated_calls(self):
        """CPU fallback returns the cached dict on every call."""
        result = _fake_result()
        pt = PendingTransfer(_event=None, _pinned_buf=None, _cpu_result=result, _buf_released=None)
        assert pt.resolve() is pt.resolve()

    def test_buf_released_event_is_set_after_resolve(self):
        """resolve() must signal the buffer release event.

        Without this, the engine's _pinned_released[idx].wait() blocks
        forever when it tries to reuse the buffer slot, deadlocking the
        pipeline after _num_pinned frames.
        """
        released = threading.Event()
        assert not released.is_set()

        mock_event = MagicMock()
        mock_event.synchronize.return_value = None
        bulk = np.zeros((4, 4, 11), dtype=np.float32)
        pinned_buf = MagicMock()
        pinned_buf.numpy.return_value = bulk

        pt = PendingTransfer(
            _event=mock_event,
            _pinned_buf=pinned_buf,
            _cpu_result=None,
            _buf_released=released,
        )
        pt.resolve()

        assert released.is_set(), "Buffer release event must be set after resolve()"
        mock_event.synchronize.assert_called_once()

    def test_gpu_bulk_released_after_resolve(self):
        """resolve() must clear _gpu_bulk after DMA sync.

        Holding _gpu_bulk prevents the CUDA caching allocator from reusing
        the GPU memory while the DMA is still reading from it (the root
        cause of intermittent bottom-of-frame banding corruption).
        Clearing it after sync allows the allocator to reclaim the memory.
        """
        mock_event = MagicMock()
        mock_event.synchronize.return_value = None
        bulk = np.zeros((4, 4, 11), dtype=np.float32)
        pinned_buf = MagicMock()
        pinned_buf.numpy.return_value = bulk
        gpu_bulk = MagicMock()

        pt = PendingTransfer(
            _event=mock_event,
            _pinned_buf=pinned_buf,
            _cpu_result=None,
            _buf_released=threading.Event(),
            _gpu_bulk=gpu_bulk,
        )
        assert pt._gpu_bulk is not None
        pt.resolve()
        assert pt._gpu_bulk is None, "_gpu_bulk must be released after DMA sync"

    def test_resolve_copies_out_of_pinned_buffer(self):
        """resolve() must .copy() slices so the result doesn't alias the pinned buffer.

        Without copies, the next frame's DMA overwrites the previous
        frame's result data while the writer is still encoding it to disk.
        """
        mock_event = MagicMock()
        mock_event.synchronize.return_value = None
        bulk = np.ones((4, 4, 11), dtype=np.float32)
        pinned_buf = MagicMock()
        pinned_buf.numpy.return_value = bulk

        pt = PendingTransfer(
            _event=mock_event,
            _pinned_buf=pinned_buf,
            _cpu_result=None,
            _buf_released=threading.Event(),
        )
        result = pt.resolve()

        # Mutate the pinned buffer — result must not change
        bulk[:] = 999.0
        assert result["alpha"].max() == 1.0, "Result must be a copy, not a view of the pinned buffer"


# ---------------------------------------------------------------------------
# Drain worker sentinel cascade
# ---------------------------------------------------------------------------


class TestDrainWorkerLifecycle:
    """Verify drain worker shutdown via sentinel cascade.

    The pipeline puts a single None on write_q after inference threads
    finish. Each drain worker that sees None puts it back before exiting,
    so the next worker sees it too. This must reliably stop all N workers.
    """

    def test_sentinel_cascade_stops_all_workers(self):
        """A single None sentinel must cascade to stop N drain workers."""
        import queue as _queue

        write_q: _queue.Queue = _queue.Queue()
        stopped = []
        lock = threading.Lock()

        def drain_worker(worker_id: int) -> None:
            while True:
                item = write_q.get()
                if item is None:
                    write_q.put(None)  # cascade
                    with lock:
                        stopped.append(worker_id)
                    break

        workers = [threading.Thread(target=drain_worker, args=(i,)) for i in range(4)]
        for w in workers:
            w.start()

        write_q.put(None)

        for w in workers:
            w.join(timeout=5)
            assert not w.is_alive(), "Drain worker should have stopped"

        assert sorted(stopped) == [0, 1, 2, 3]

    def test_drain_processes_all_items_before_stopping(self):
        """Items queued before the sentinel must all be processed."""
        import queue as _queue

        write_q: _queue.Queue = _queue.Queue()
        processed = []
        lock = threading.Lock()

        def drain_worker() -> None:
            while True:
                item = write_q.get()
                if item is None:
                    write_q.put(None)
                    break
                with lock:
                    processed.append(item)

        for i in range(10):
            write_q.put(i)
        write_q.put(None)

        workers = [threading.Thread(target=drain_worker) for _ in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=5)

        assert sorted(processed) == list(range(10))
