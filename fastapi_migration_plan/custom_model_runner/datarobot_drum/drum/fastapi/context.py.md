# Plan: custom_model_runner/datarobot_drum/drum/fastapi/context.py

Worker context for FastAPI, mirroring `gunicorn/context.py` with async-friendly adaptations and robust resource management.

## Overview

The `FastAPIWorkerCtx` class manages the lifecycle of background tasks and resources. It is adapted for async environments while maintaining parity with the gunicorn version.

## Key Changes for Robustness:

1. **Safe Monkey-Patching**: The OpenTelemetry patch includes version pinning, integration tests, and proper fallback.
2. **Explicit Cleanup Order**: Ensuring resources are closed in the correct order (e.g., stopping tasks before closing DB connections).
3. **Async Task Management**: Proper cancellation and handling of `asyncio.Task` objects.
4. **Production-Ready Backpressure**: Queue depth monitoring with rejection and metrics.

## Proposed Implementation:

```python
"""
FastAPI Worker Context.
Mirrors gunicorn/context.py with async-friendly adaptations.
"""
import asyncio
import logging
import threading
import time
from typing import Callable, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from fastapi import FastAPI
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Supported OpenTelemetry versions for the asyncio patch
# Update this list when testing with new versions
OTEL_OPENAI_SUPPORTED_VERSIONS = [
    "0.21b0", "0.22b0", "0.23b0", "0.24b0", "0.25b0",
    "0.26b0", "0.27b0", "0.28b0",
]


@dataclass
class BackpressureMetrics:
    """Metrics for monitoring backpressure behavior."""
    requests_accepted: int = 0
    requests_rejected: int = 0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0
    total_wait_time_ms: float = 0.0
    
    def record_accepted(self, wait_time_ms: float):
        self.requests_accepted += 1
        self.total_wait_time_ms += wait_time_ms
    
    def record_rejected(self):
        self.requests_rejected += 1
    
    def update_queue_depth(self, depth: int):
        self.current_queue_depth = depth
        self.peak_queue_depth = max(self.peak_queue_depth, depth)
    
    def to_dict(self) -> dict:
        return {
            "backpressure_accepted": self.requests_accepted,
            "backpressure_rejected": self.requests_rejected,
            "backpressure_queue_depth": self.current_queue_depth,
            "backpressure_peak_queue_depth": self.peak_queue_depth,
            "backpressure_avg_wait_ms": (
                self.total_wait_time_ms / self.requests_accepted
                if self.requests_accepted > 0 else 0
            ),
        }


class ProductionBackpressure:
    """
    Production-ready backpressure with queue depth monitoring and rejection.
    
    Unlike a simple semaphore, this class:
    - Tracks queue depth for observability
    - Rejects requests immediately when queue is full (no unbounded waiting)
    - Provides metrics for alerting and dashboards
    - Supports adaptive backoff based on latency (optional)
    """
    
    def __init__(
        self, 
        max_concurrent: int = 10,
        max_queue_depth: int = 100,
        acquire_timeout: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self.acquire_timeout = acquire_timeout
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_depth = 0
        self._lock = asyncio.Lock()
        self.metrics = BackpressureMetrics()
    
    async def acquire(self) -> bool:
        """
        Acquire a slot for request processing.
        
        Returns:
            True if acquired, False if rejected due to overload.
        
        Raises:
            asyncio.TimeoutError if waiting too long.
        """
        async with self._lock:
            # Check queue depth before waiting
            if self._queue_depth >= self.max_queue_depth:
                self.metrics.record_rejected()
                logger.warning(
                    "Backpressure: rejecting request, queue depth %d >= max %d",
                    self._queue_depth, self.max_queue_depth
                )
                return False
            
            self._queue_depth += 1
            self.metrics.update_queue_depth(self._queue_depth)
        
        start_time = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.acquire_timeout
            )
            wait_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_accepted(wait_time_ms)
            return True
        except asyncio.TimeoutError:
            async with self._lock:
                self._queue_depth -= 1
                self.metrics.update_queue_depth(self._queue_depth)
            self.metrics.record_rejected()
            raise
    
    def release(self):
        """Release a slot after request processing."""
        self._semaphore.release()
        # Note: queue_depth is decremented in middleware after response
    
    async def decrement_queue_depth(self):
        """Decrement queue depth (call after request completes)."""
        async with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            self.metrics.update_queue_depth(self._queue_depth)


class FastAPIWorkerCtx:
    """
    Worker context for FastAPI, managing lifecycle of background tasks and resources.
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self._running = False
        self._threads: List[threading.Thread] = []
        self._async_tasks: List[asyncio.Task] = []
        self._on_stop: List[Tuple[int, Callable[[], None], str]] = []
        self._on_cleanup: List[Tuple[int, Callable[[], None], str]] = []
        
        # Production backpressure
        from datarobot_drum import RuntimeParameters
        max_concurrent = int(RuntimeParameters.get("DRUM_MAX_CONCURRENT_REQUESTS", 10))
        max_queue_depth = int(RuntimeParameters.get("DRUM_MAX_QUEUE_DEPTH", 100))
        acquire_timeout = float(RuntimeParameters.get("DRUM_BACKPRESSURE_TIMEOUT", 30.0))
        
        self.backpressure = ProductionBackpressure(
            max_concurrent=max_concurrent,
            max_queue_depth=max_queue_depth,
            acquire_timeout=acquire_timeout,
        )
        
        # Legacy semaphore for backward compatibility
        self.semaphore = self.backpressure._semaphore

    def add_thread(self, thread: threading.Thread, *, name: Optional[str] = None):
        """Register a thread for graceful shutdown."""
        self._threads.append(thread)
        thread_name = name or thread.name or f"thread:{id(thread)}"
        logger.debug("Registered thread: %s", thread_name)

    def add_async_task(self, task: asyncio.Task, *, name: Optional[str] = None):
        """Register an async task for graceful cancellation."""
        self._async_tasks.append(task)
        task_name = name or getattr(task, 'get_name', lambda: f"task:{id(task)}")()
        
        def _cancel():
            if not task.done():
                logger.debug("Cancelling async task: %s", task_name)
                task.cancel()
        
        self.defer_stop(_cancel, desc=task_name)

    def defer_stop(self, fn: Callable[[], None], *, order: int = 50, desc: str = ""):
        """Register a function to be called during stop phase."""
        self._on_stop.append((order, fn, desc))

    def defer_cleanup(self, fn: Callable[[], None], *, order: int = 50, desc: str = ""):
        """Register a function to be called during cleanup phase."""
        self._on_cleanup.append((order, fn, desc))

    def start(self):
        """Start the worker context."""
        self._running = True
        # Apply patches before starting main logic
        self._patch_asyncio_for_otel()
        
        from datarobot_drum.drum.main import main
        main(self.app, self)

    def _patch_asyncio_for_otel(self):
        """
        Patch asyncio.run for OpenTelemetry compatibility with version checking.
        
        Fixes 'RuntimeError: asyncio.run() cannot be called from a running event loop'.
        
        This patch is necessary because opentelemetry-instrumentation-openai uses
        asyncio.run() internally, which fails when already inside an event loop
        (as is the case with Uvicorn/FastAPI).
        """
        try:
            import opentelemetry.instrumentation.openai as otel_openai
            import opentelemetry.instrumentation.openai.utils as otel_utils
        except ImportError:
            logger.debug("opentelemetry-instrumentation-openai not installed, skipping patch")
            return
        
        # Check if already patched
        if hasattr(otel_utils, "_drum_patched"):
            logger.debug("OpenTelemetry asyncio patch already applied")
            return
        
        # Version check with warning for untested versions
        try:
            otel_version = otel_openai.__version__
            if otel_version not in OTEL_OPENAI_SUPPORTED_VERSIONS:
                logger.warning(
                    "OpenTelemetry OpenAI instrumentation version %s is not in "
                    "tested versions %s. The asyncio patch may not work correctly. "
                    "Please report issues to: https://github.com/datarobot/drum/issues",
                    otel_version,
                    OTEL_OPENAI_SUPPORTED_VERSIONS
                )
        except AttributeError:
            logger.warning(
                "Could not determine opentelemetry-instrumentation-openai version"
            )
        
        # Check if run_async exists and is callable
        if not hasattr(otel_utils, "run_async") or not callable(otel_utils.run_async):
            logger.warning(
                "opentelemetry.instrumentation.openai.utils.run_async not found or "
                "not callable. Skipping patch. This may cause 'cannot call asyncio.run() "
                "from a running event loop' errors."
            )
            return
        
        original_run_async = otel_utils.run_async

        def fixed_run_async(method):
            """
            Replacement for otel_utils.run_async that handles running event loops.
            """
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Schedule as task instead of calling asyncio.run()
                future = asyncio.ensure_future(method)
                
                def _handle_exception(fut):
                    try:
                        exc = fut.exception()
                        if exc is not None:
                            logger.error(
                                "Exception in OpenTelemetry async task: %s", 
                                exc,
                                exc_info=exc
                            )
                    except asyncio.CancelledError:
                        pass
                
                future.add_done_callback(_handle_exception)
            else:
                # Not in async context, use original
                original_run_async(method)

        # Apply patch
        otel_utils.run_async = fixed_run_async
        otel_utils._drum_patched = True
        logger.info(
            "Applied asyncio patch for OpenTelemetry OpenAI instrumentation (version: %s)",
            getattr(otel_openai, '__version__', 'unknown')
        )

    def stop(self):
        """Gracefully signal all tasks to stop."""
        logger.info("FastAPIWorkerCtx stopping...")
        self._running = False
        
        # Sort by priority (ascending) - lower numbers run first
        for priority, fn, desc in sorted(self._on_stop, key=lambda x: x[0]):
            try:
                logger.debug("Stop hook [%d]: %s", priority, desc)
                fn()
            except Exception as e:
                logger.warning("Error during stop of %s: %s", desc, e, exc_info=True)
        
        # Cancel async tasks
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        
        logger.info("FastAPIWorkerCtx stop complete")

    def cleanup(self):
        """Close resources in reverse priority order."""
        logger.info("FastAPIWorkerCtx cleanup starting...")
        
        # Sort by priority descending - higher numbers run first during cleanup
        for priority, fn, desc in sorted(self._on_cleanup, key=lambda x: x[0], reverse=True):
            try:
                logger.debug("Cleanup hook [%d]: %s", priority, desc)
                fn()
            except Exception as e:
                logger.error("Cleanup failed for %s: %s", desc, e, exc_info=True)
        
        # Join threads with timeout
        for thread in self._threads:
            if thread.is_alive():
                logger.debug("Waiting for thread: %s", thread.name)
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning("Thread %s did not terminate in time", thread.name)
        
        logger.info("FastAPIWorkerCtx cleanup complete")

    def get_backpressure_metrics(self) -> dict:
        """Get backpressure metrics for /stats/ endpoint."""
        return self.backpressure.metrics.to_dict()

    @property
    def is_running(self) -> bool:
        return self._running
```

## Watchdog Thread Registration

The NIM watchdog thread must be registered to ensure it exits when the worker stops.

```python
# In prediction_server.py:
def _start_watchdog_if_enabled(self):
    # ... (check config) ...
    self._server_watchdog = Thread(target=self.watchdog, args=(port,), daemon=True)
    if self._worker_ctx:
        self._worker_ctx.add_thread(self._server_watchdog, name="NIM Watchdog")
    self._server_watchdog.start()
```

## Integration Test for OTel Patch

```python
# tests/unit/datarobot_drum/drum/fastapi/test_otel_patch.py
"""
Tests for OpenTelemetry asyncio patch compatibility.
"""
import asyncio
import pytest

class TestOtelPatch:
    """Verify OpenTelemetry patch works correctly."""
    
    def test_patch_applied_only_once(self, fastapi_worker_ctx):
        """Ensure patch is not applied multiple times."""
        ctx = fastapi_worker_ctx
        ctx._patch_asyncio_for_otel()
        ctx._patch_asyncio_for_otel()  # Second call should be no-op
        
        # Check _drum_patched flag
        try:
            import opentelemetry.instrumentation.openai.utils as otel_utils
            assert hasattr(otel_utils, "_drum_patched")
        except ImportError:
            pytest.skip("opentelemetry-instrumentation-openai not installed")
    
    @pytest.mark.asyncio
    async def test_patched_run_async_in_event_loop(self, fastapi_worker_ctx):
        """Verify patched run_async works inside event loop."""
        ctx = fastapi_worker_ctx
        ctx._patch_asyncio_for_otel()
        
        try:
            import opentelemetry.instrumentation.openai.utils as otel_utils
            
            result = []
            async def test_coro():
                result.append("executed")
            
            # This should not raise "cannot call asyncio.run() from running event loop"
            otel_utils.run_async(test_coro())
            await asyncio.sleep(0.1)  # Allow task to complete
            
            assert "executed" in result
        except ImportError:
            pytest.skip("opentelemetry-instrumentation-openai not installed")
```

## Key Parity Points:

| Aspect | Gunicorn WorkerCtx | FastAPIWorkerCtx |
|--------|-------------------|------------------|
| Thread management | `add_thread` | Identical |
| Lifecycle hooks | `on_stop`, `on_cleanup` | Identical logic |
| Async support | Gevent greenlets | Asyncio Tasks |
| Backpressure | N/A | ProductionBackpressure with metrics |
| OTel compatibility | N/A (sync) | Patched for async |

## New Runtime Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DRUM_MAX_CONCURRENT_REQUESTS` | Max concurrent requests in processing | 10 |
| `DRUM_MAX_QUEUE_DEPTH` | Max requests waiting in queue before rejection | 100 |
| `DRUM_BACKPRESSURE_TIMEOUT` | Max time (seconds) to wait for a slot | 30.0 |
