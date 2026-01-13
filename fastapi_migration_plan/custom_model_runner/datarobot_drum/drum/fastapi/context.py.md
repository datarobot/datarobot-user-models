# Plan: custom_model_runner/datarobot_drum/drum/fastapi/context.py

Worker context for FastAPI, mirroring `gunicorn/context.py` with async-friendly adaptations.

## Overview

The `FastAPIWorkerCtx` class manages the lifecycle of background tasks and resources within a FastAPI/Uvicorn worker. It provides the same interface as the gunicorn `WorkerCtx` but is adapted for async environments.

## Proposed Implementation:

```python
"""
FastAPI Worker Context.
Mirrors gunicorn/context.py with async-friendly adaptations.
"""
import asyncio
import logging
import threading
from typing import Callable, Any, List, Tuple, Optional

from fastapi import FastAPI

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class FastAPIWorkerCtx:
    """
    Context of a single Uvicorn worker:
    - .start() — start background tasks and DRUM runtime
    - .stop()  — stop background tasks (graceful)
    - .cleanup() — close resources (DB/clients) in the correct order
    - add_* methods for registering objects/callbacks for stopping/cleaning up
    
    This class mirrors WorkerCtx from gunicorn/context.py but is adapted
    for FastAPI/Uvicorn's async environment.
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self._running = False
        self._threads: List[threading.Thread] = []
        self._async_tasks: List[asyncio.Task] = []
        self._on_stop: List[Tuple[int, Callable[[], None], str]] = []
        self._on_cleanup: List[Tuple[int, Callable[[], None], str]] = []

    def add_thread(
        self, t: threading.Thread, *, join_timeout: float = 2.0, name: Optional[str] = None
    ):
        """
        Adds a thread to the worker context and registers it for graceful stopping.

        Args:
            t (threading.Thread): The thread to be added.
            join_timeout (float, optional): The timeout for joining the thread during stop. Defaults to 2.0 seconds.
            name (Optional[str], optional): A descriptive name for the thread. Defaults to None.
        """
        t.daemon = True
        self._threads.append(t)

        def _join():
            if t.is_alive():
                t.join(join_timeout)

        self.defer_stop(_join, desc=name or f"thread:{id(t)}")

    def add_async_task(
        self, task: asyncio.Task, *, cancel_timeout: float = 2.0, name: Optional[str] = None
    ):
        """
        Adds an async task to the worker context and registers it for graceful stopping.
        
        Note: This replaces add_greenlet() from gunicorn context since Uvicorn uses asyncio.

        Args:
            task (asyncio.Task): The async task to be added.
            cancel_timeout (float, optional): The timeout for cancelling the task during stop. Defaults to 2.0 seconds.
            name (Optional[str], optional): A descriptive name for the task. Defaults to None.
        """
        self._async_tasks.append(task)

        def _cancel():
            if not task.done():
                task.cancel()
                try:
                    # Give the task time to cleanup
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't await in sync context, just cancel
                        pass
                except Exception:
                    pass

        self.defer_stop(_cancel, desc=name or f"async_task:{id(task)}")

    def add_closeable(
        self, obj: Any, method: str = "close", *, order: int = 0, desc: Optional[str] = None
    ):
        """
        Registers an object to be closed using a specified method during cleanup.

        Args:
            obj (Any): The object to be closed.
            method (str, optional): The method to call on the object for closing. Defaults to "close".
            order (int, optional): The priority order for cleanup. Lower values are executed later. Defaults to 0.
            desc (Optional[str], optional): A description for the cleanup action. Defaults to None.
        """
        fn = getattr(obj, method, None)
        if callable(fn):
            self.defer_cleanup(fn, order=order, desc=desc or f"{obj}.{method}")

    def defer_stop(self, fn: Callable[[], None], *, order: int = 0, desc: str = "on_stop"):
        """Register a callback to be called during stop()."""
        self._on_stop.append((order, fn, desc))

    def defer_cleanup(self, fn: Callable[[], None], *, order: int = 0, desc: str = "on_cleanup"):
        """Register a callback to be called during cleanup()."""
        self._on_cleanup.append((order, fn, desc))

    def start(self):
        """
        Starts background tasks for the worker context.

        This method sets the running flag to True and initializes the main application logic.
        It imports and calls the `main` function from `datarobot_drum.drum.main`, passing
        the application instance and the worker context as arguments.
        """
        self._running = True

        # Fix RuntimeError: asyncio.run() cannot be called from a running event loop
        # This is needed for OpenTelemetry instrumentation
        self._patch_asyncio_for_otel()

        from datarobot_drum.drum.main import main
        main(self.app, self)

    def _patch_asyncio_for_otel(self):
        """
        Patch asyncio.run for OpenTelemetry compatibility.
        
        When running inside Uvicorn, there's already an event loop running.
        OpenTelemetry's async instrumentation may call asyncio.run() which fails
        in this case. This patch makes it work by scheduling coroutines on the
        existing loop instead.
        """
        try:
            import opentelemetry.instrumentation.openai.utils as otel_utils
        except ImportError:
            return  # OpenTelemetry not installed or openai instrumentation not available
        
        def fixed_run_async(method):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                def handle_task_result(task):
                    try:
                        task.result()  # retrieve exception if any
                    except Exception:
                        logger.error("OpenTelemetry async task error")

                # Schedule the coroutine safely and add done callback to handle errors
                task = loop.create_task(method)
                task.add_done_callback(handle_task_result)
            else:
                asyncio.run(method)

        # Apply monkey patch
        otel_utils.run_async = fixed_run_async
        logger.debug("Applied asyncio patch for OpenTelemetry compatibility")

    def stop(self):
        """
        Gracefully stops the worker context.

        This method sets the running flag to False and executes all registered
        stop callbacks in the order of their priority.
        """
        self._running = False
        for _, fn, desc in sorted(self._on_stop, key=lambda x: x[0]):
            try:
                logger.debug("Stopping: %s", desc)
                fn()
            except Exception as e:
                logger.warning("Error during stop of %s: %s", desc, e)

    def cleanup(self):
        """
        Cleans up resources in the worker context.

        This method is called at the end of the worker's lifecycle to release resources.
        It executes all registered cleanup callbacks in reverse priority order.
        """
        for _, fn, desc in sorted(self._on_cleanup, key=lambda x: x[0], reverse=True):
            try:
                logger.info("FastAPIWorkerCtx cleanup: %s", desc)
                fn()
            except Exception as e:
                logger.error("Cleanup failed for %s: %s", desc, e)

    def running(self) -> bool:
        """Check if the worker is running."""
        return self._running


def create_ctx(app: FastAPI) -> FastAPIWorkerCtx:
    """
    Factory method to create a FastAPIWorkerCtx instance.

    This method initializes the worker context for the application and allows
    registration of objects or callbacks for stopping and cleanup.

    Args:
        app: The FastAPI application instance.

    Returns:
        FastAPIWorkerCtx: The initialized worker context.
    """
    ctx = FastAPIWorkerCtx(app)
    return ctx
```

## Watchdog Integration

The NIM watchdog thread monitors server health and forcefully terminates processes if health checks fail. This needs to be integrated with the FastAPI worker context.

### Watchdog Thread Registration

In `prediction_server.py`, register the watchdog thread with the worker context:

```python
from threading import Thread
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.enum import URL_PREFIX_ENV_VAR_NAME

class PredictionServer(PredictMixin):
    
    def _start_watchdog(self, port: int, worker_ctx: "FastAPIWorkerCtx" = None):
        """
        Start the NIM watchdog thread if enabled.
        
        Args:
            port: Server port for health checks
            worker_ctx: Optional worker context to register the thread for cleanup
        """
        if not RuntimeParameters.has("USE_NIM_WATCHDOG"):
            return
        
        if str(RuntimeParameters.get("USE_NIM_WATCHDOG")).lower() not in ["true", "1", "yes"]:
            return
        
        # Note: watchdog() method must check self._running flag for graceful shutdown
        self._server_watchdog = Thread(
            target=self.watchdog,
            args=(port,),
            daemon=True,
            name="NIM Sidecar Watchdog",
        )
        
        # Register with worker context for proper cleanup
        if worker_ctx is not None:
            worker_ctx.add_thread(
                self._server_watchdog,
                join_timeout=5.0,
                name="NIM Sidecar Watchdog"
            )
        
        self._server_watchdog.start()
        logger.info("Started NIM watchdog thread for port %s", port)
```

### Updated Watchdog Implementation

The `watchdog()` method in `PredictionServer` must be updated to check `_running`:

```python
    def watchdog(self, port):
        """
        Watchdog thread that periodically checks if the server is alive.
        Supports graceful shutdown by checking the _running flag.
        """
        # ... (initialization code) ...
        
        while self._running:  # CRITICAL: Check _running flag
            try:
                # ... (health check logic) ...
                if response.ok:
                    attempt = 0
                    # Sleep in small increments to respond quickly to shutdown signal
                    for _ in range(check_interval):
                        if not self._running:
                            return
                        time.sleep(1)
                    continue
                # ...
            except Exception as e:
                # ...
```

## Resource Cleanup Order

Cleanup is performed in reverse priority order (higher `order` values are executed first during cleanup).

| Resource | Order | Description |
|----------|-------|-------------|
| Predictor / Model | 200 | Shutdown model-specific resources first |
| ThreadPoolExecutor | 100 | Wait for active prediction threads to finish |
| StdoutFlusher | 50 | Stop flusher thread after predictions are done |
| DB / External Clients | 0 | Close network connections last |

In `FastAPIWorkerCtx.cleanup()`:
```python
    def cleanup(self):
        """
        Cleans up resources in the worker context.
        Executes all registered cleanup callbacks in reverse priority order.
        """
        # Sort by order DESCENDING (higher order first)
        for _, fn, desc in sorted(self._on_cleanup, key=lambda x: x[0], reverse=True):
            try:
                logger.info("FastAPIWorkerCtx cleanup: %s", desc)
                fn()
            except Exception as e:
                logger.error("Cleanup failed for %s: %s", desc, e)
```

## Key Differences from Gunicorn WorkerCtx

| Aspect | Gunicorn WorkerCtx | FastAPIWorkerCtx |
|--------|-------------------|------------------|
| App type | Flask | FastAPI |
| Green threads | `add_greenlet()` with gevent | `add_async_task()` with asyncio |
| Event loop | N/A (sync) | asyncio event loop |
| OTel patch | In `start()` method | In `_patch_asyncio_for_otel()` |

## Notes:
- The `add_greenlet()` method is replaced with `add_async_task()` since Uvicorn uses asyncio instead of gevent.
- The asyncio patch for OpenTelemetry is extracted into a separate method for clarity.
- Thread management remains the same as gunicorn's WorkerCtx.
- The cleanup order (reverse priority) is preserved from the original implementation.
- Watchdog thread is registered with the worker context for proper cleanup during shutdown.
- The `_running` flag is used to signal the watchdog to exit gracefully.
