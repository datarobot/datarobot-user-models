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

### Updated Lifespan Handler

In `app.py`, start the watchdog after the predictor is initialized:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    """
    # === STARTUP ===
    logger.info("FastAPI lifespan startup initiated")
    
    # Restore sys.argv from environment variable
    drum_args = os.environ.get("DRUM_UVICORN_DRUM_ARGS", "")
    if drum_args:
        sys.argv = shlex.split(drum_args)
        logger.debug("Restored sys.argv: %s", sys.argv)
    
    os.environ["MAX_WORKERS"] = "1"
    
    from datarobot_drum import RuntimeParameters
    if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
        os.environ.pop("MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS", None)
    
    # Create worker context
    from datarobot_drum.drum.fastapi.context import create_ctx
    ctx = create_ctx(app)
    set_worker_ctx(ctx)
    
    # Start the DRUM runtime (loads model, initializes predictor, etc.)
    ctx.start()
    
    # Start watchdog after predictor is initialized
    # The prediction_server should call _start_watchdog() with ctx
    
    logger.info("FastAPI lifespan startup complete")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    logger.info("FastAPI lifespan shutdown initiated")
    
    ctx = get_worker_ctx()
    if ctx:
        try:
            # This will stop all registered threads including watchdog
            ctx.stop()
        except Exception as e:
            logger.error("Error during context stop: %s", e)
        finally:
            ctx.cleanup()
    
    logger.info("FastAPI lifespan shutdown complete")
```

### Watchdog Health Check URL for FastAPI

Update the watchdog to use the correct health URL:

```python
def watchdog(self, port):
    """
    Watchdog thread that periodically checks if the server is alive.
    Works with both Flask and FastAPI servers.
    """
    import os
    import requests
    import time
    
    logger.info("Starting watchdog to monitor server health...")
    
    url_host = os.environ.get("TEST_URL_HOST", "localhost")
    url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
    
    # Use /info/ endpoint for health check (works with both Flask and FastAPI)
    health_url = f"http://{url_host}:{port}{url_prefix}/info/"
    
    request_timeout = 120
    if RuntimeParameters.has("NIM_WATCHDOG_REQUEST_TIMEOUT"):
        try:
            request_timeout = int(RuntimeParameters.get("NIM_WATCHDOG_REQUEST_TIMEOUT"))
        except ValueError:
            logger.warning("Invalid NIM_WATCHDOG_REQUEST_TIMEOUT, using default 120s")
    
    check_interval = 10  # seconds
    max_attempts = 3
    
    if RuntimeParameters.has("NIM_WATCHDOG_MAX_ATTEMPTS"):
        try:
            max_attempts = int(RuntimeParameters.get("NIM_WATCHDOG_MAX_ATTEMPTS"))
        except ValueError:
            logger.warning("Invalid NIM_WATCHDOG_MAX_ATTEMPTS, using default 3")
    
    attempt = 0
    base_sleep_time = 4
    
    while self._running if hasattr(self, '_running') else True:
        try:
            logger.debug("Server health check: %s", health_url)
            response = requests.get(health_url, timeout=request_timeout)
            logger.debug("Server health check status: %s", response.status_code)
            
            if response.ok:
                attempt = 0
                time.sleep(check_interval)
                continue
            else:
                raise Exception(f"Health check returned {response.status_code}")
                
        except Exception as e:
            attempt += 1
            logger.warning(
                "Server health check failed (attempt %d/%d): %s",
                attempt, max_attempts, str(e)
            )
            
            if attempt >= max_attempts:
                self._kill_all_processes()
                return  # Exit watchdog after killing processes
            
            # Quadratic backoff
            sleep_time = base_sleep_time * (attempt ** 2)
            logger.info("Retrying in %d seconds...", sleep_time)
            time.sleep(sleep_time)
```

### Graceful Watchdog Shutdown

The watchdog should check the `_running` flag to exit gracefully:

```python
class FastAPIWorkerCtx:
    # ... existing code ...
    
    def add_watchdog_thread(
        self, 
        watchdog_fn: Callable[[int], None],
        port: int,
        name: str = "Watchdog"
    ):
        """
        Special method to add a watchdog thread that checks the running flag.
        
        Args:
            watchdog_fn: The watchdog function to run
            port: Server port for health checks
            name: Thread name
        """
        # Create a wrapper that passes the running check
        def watchdog_wrapper():
            while self._running:
                try:
                    watchdog_fn(port)
                except Exception as e:
                    logger.error("Watchdog error: %s", e)
                    break
        
        t = threading.Thread(
            target=watchdog_wrapper,
            daemon=True,
            name=name,
        )
        self.add_thread(t, join_timeout=5.0, name=name)
        t.start()
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
