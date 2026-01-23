import logging
import threading
from typing import Callable, Any, List, Tuple, Optional

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class WorkerCtx:
    """
    Context of a single gunicorn worker:
    - .start() — start background tasks
    - .stop()  — stop background tasks (graceful)
    - .cleanup() — close resources (DB/clients) in the correct order
    - add_* methods for registering objects/callbacks for stopping/cleaning up
    """

    def __init__(self, app):
        self.app = app
        self._running = False
        self._threads: List[threading.Thread] = []
        self._greenlets: List[Any] = []
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

    def add_greenlet(self, g, *, kill_timeout: float = 2.0, name: Optional[str] = None):
        """
        Adds a greenlet to the worker context and registers it for graceful stopping.

        Args:
            g: The greenlet to be added.
            kill_timeout (float, optional): The timeout for killing the greenlet during stop. Defaults to 2.0 seconds.
            name (Optional[str], optional): A descriptive name for the greenlet. Defaults to None.
        """
        self._greenlets.append(g)

        def _kill():
            try:
                g.kill(block=True, timeout=kill_timeout)
            except Exception:
                pass

        self.defer_stop(_kill, desc=name or f"greenlet:{id(g)}")

    def add_closeable(
        self, obj: Any, method: str = "close", *, order: int = 0, desc: Optional[str] = None
    ):
        """
        Registers an object to be closed using a specified method (e.g., close/quit/disconnect/shutdown) during cleanup (after stop).

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
        self._on_stop.append((order, fn, desc))

    def defer_cleanup(self, fn: Callable[[], None], *, order: int = 0, desc: str = "on_cleanup"):
        self._on_cleanup.append((order, fn, desc))

    def _setup_async_patches(self):
        """
        Sets up monkey patches for async libraries to ensure thread-safe execution
        in Gunicorn workers (sync/gevent).

        This method creates a dedicated background event loop and patches:
        - AsyncHTTPClient (datarobot_dome): bulk_upload_custom_metrics, predict, async_report_event
        - OpenTelemetry instrumentation: run_async
        - GuardExecutor (datarobot_dome): async_guard_executor

        All async operations are offloaded to the background loop via run_coroutine_threadsafe,
        preventing 'attached to a different loop' and 'non-thread-safe operation' errors.
        """
        import asyncio

        def _wrap_future(fut):
            """Wrap concurrent.futures.Future as asyncio.Future if inside a running loop."""
            try:
                asyncio.get_running_loop()
                return asyncio.wrap_future(fut)
            except RuntimeError:
                return fut

        # Background loop for thread-safe async execution in sync/gevent workers.
        # We store it on self.app to ensure it lives as long as the worker.
        # Guard both loop creation AND patching to prevent nested wrappers if start() is called multiple times.
        if hasattr(self.app, "_drum_bg_loop"):
            return  # Already initialized - skip to avoid double-patching

        loop = asyncio.new_event_loop()
        bg_thread = threading.Thread(target=loop.run_forever, daemon=True, name="DrumBgLoop")
        bg_thread.start()
        self.app._drum_bg_loop = loop
        self.app._drum_bg_thread = bg_thread  # Store thread reference for cleanup

        bg_loop = self.app._drum_bg_loop

        # Register cleanup for the background loop to allow pending async operations
        # (metrics uploads, guard executions, telemetry) to complete during shutdown.
        def _cleanup_bg_loop():
            # Capture thread reference in closure to avoid variable shadowing issues
            loop_thread = self.app._drum_bg_thread

            # Check if loop is still running before attempting graceful shutdown
            if not bg_loop.is_running():
                logger.debug("Background loop already stopped, skipping graceful shutdown")
                # Still need to join thread and close loop
                if loop_thread.is_alive():
                    loop_thread.join(timeout=2.0)
                try:
                    bg_loop.close()
                except Exception:
                    pass
                return

            try:
                # Give pending tasks a chance to complete by scheduling a graceful shutdown
                async def _shutdown():
                    tasks = [
                        task
                        for task in asyncio.all_tasks(bg_loop)
                        if task is not asyncio.current_task()
                    ]
                    if tasks:
                        # Wait up to 5 seconds for pending tasks to complete
                        done, pending = await asyncio.wait(tasks, timeout=5.0)
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                    bg_loop.stop()

                asyncio.run_coroutine_threadsafe(_shutdown(), bg_loop).result(timeout=10.0)
            except (KeyboardInterrupt, SystemExit) as e:
                # Handle interruption signals during shutdown - force stop but don't re-raise
                # to allow other cleanup callbacks to complete. Gunicorn handles signals at a higher level.
                logger.warning("Interrupted during background loop shutdown: %s", type(e).__name__)
                try:
                    bg_loop.call_soon_threadsafe(bg_loop.stop)
                except RuntimeError:
                    pass
            except Exception as e:
                logger.warning("Error during background loop shutdown: %s", e)
                # Force stop if graceful shutdown fails
                try:
                    bg_loop.call_soon_threadsafe(bg_loop.stop)
                except RuntimeError:
                    # Loop may already be closed
                    pass
            finally:
                # Join the thread with a timeout
                if loop_thread.is_alive():
                    loop_thread.join(timeout=2.0)
                    if loop_thread.is_alive():
                        logger.warning("Background loop thread did not terminate within timeout")
                # Close the loop
                try:
                    bg_loop.close()
                except Exception as e:
                    logger.debug("Error closing background loop: %s", e)

        # Use low order (negative) so loop cleanup happens after other cleanup callbacks
        # that might still need to use async operations
        self.defer_cleanup(_cleanup_bg_loop, order=-100, desc="background event loop")

        self._patch_async_http_client(bg_loop, _wrap_future)
        self._patch_otel_utils(bg_loop, _wrap_future)
        self._patch_guard_executor(bg_loop, _wrap_future)

    def _patch_async_http_client(self, bg_loop, _wrap_future):
        """Patch AsyncHTTPClient methods to run in the background loop."""
        import asyncio
        import weakref

        try:
            from datarobot_dome.async_http_client import AsyncHTTPClient
        except ImportError:
            return

        # Per-instance locks for session recreation.
        # Using WeakKeyDictionary to automatically remove entries when client is garbage collected,
        # preventing memory leaks in long-running workers with dynamic client creation.
        _session_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # Threading lock to protect the check-then-set pattern in _get_session_lock.
        # While currently _get_session_lock is only called from coroutines on bg_loop,
        # this makes the code defensive against future changes that might call it from other threads.
        _session_locks_guard = threading.Lock()

        def _get_session_lock(client) -> asyncio.Lock:
            """Get or create an asyncio.Lock for the given client instance (thread-safe)."""
            with _session_locks_guard:
                if client not in _session_locks:
                    _session_locks[client] = asyncio.Lock()
                return _session_locks[client]

        def patch_method(method_name):
            _orig_method = getattr(AsyncHTTPClient, method_name)

            def fixed_method(self, *args, **kwargs):
                """
                Thread-safe wrapper that offloads the async execution to a dedicated background loop.
                This prevents 'attached to a different loop' and 'non-thread-safe operation' errors.
                """

                def _get_session_loop(session):
                    """
                    Extract the event loop from an aiohttp session.
                    Checks multiple locations for compatibility with different aiohttp versions:
                    - session._loop: older aiohttp versions (< 3.8)
                    - session._connector._loop: aiohttp 3.8+
                    - session.connector._loop: aiohttp 3.9+ (connector is public property)

                    Note: This relies on internal aiohttp attributes which may change.
                    If all methods fail, returns None and session will be recreated.
                    """
                    # Try direct _loop attribute (older versions < 3.8)
                    loop = getattr(session, "_loop", None)
                    if loop is not None:
                        return loop

                    # Try connector's loop - first via private attribute (aiohttp 3.8+)
                    connector = getattr(session, "_connector", None)
                    if connector is not None:
                        loop = getattr(connector, "_loop", None)
                        if loop is not None:
                            return loop

                    # Try public connector property (aiohttp 3.9+)
                    try:
                        connector = session.connector
                        if connector is not None:
                            loop = getattr(connector, "_loop", None)
                            if loop is not None:
                                return loop
                    except Exception:
                        pass

                    # Fallback: if we can't determine the loop, return None
                    # The session will be recreated to ensure correct loop affinity
                    logger.debug(
                        "Could not determine session's event loop, will recreate session"
                    )
                    return None

                async def _coro():
                    # 1. Ensure the client's internal loop pointer matches our background loop
                    for attr in ["loop", "_loop"]:
                        if hasattr(self, attr) and getattr(self, attr) is not bg_loop:
                            setattr(self, attr, bg_loop)

                    # 2. Session affinity: aiohttp session MUST match the background loop
                    # Use a lock to prevent race conditions when multiple coroutines
                    # concurrently detect a stale session and try to recreate it.
                    # Without the lock, coroutine A could create a new session, then
                    # coroutine B (which started its check before A finished) could
                    # set self.session = None, orphaning A's newly created session.
                    async with _get_session_lock(self):
                        if hasattr(self, "session") and self.session:
                            session_loop = _get_session_loop(self.session)
                            # Recreate if loop doesn't match or if we couldn't determine the loop
                            if session_loop is None or session_loop is not bg_loop:
                                # Force recreation in the correct loop
                                try:
                                    await self.session.close()
                                except Exception as e:
                                    logger.debug("Error closing stale session: %s", e)
                                self.session = None

                        session = getattr(self, "session", None)
                        if not session or session.closed:
                            import aiohttp
                            from datarobot_drum import RuntimeParameters

                            # Use DRUM_CLIENT_REQUEST_TIMEOUT from runtime parameters (same as gunicorn.conf.py)
                            timeout = 120
                            if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
                                temp_timeout = int(
                                    RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT")
                                )
                                if 0 <= temp_timeout <= 3600:
                                    timeout = temp_timeout

                            # Set all timeout values including total to prevent hung requests.
                            # total: overall timeout for entire operation
                            # connect/sock_connect: connection establishment timeout
                            # sock_read: timeout between data chunks
                            client_timeout = aiohttp.ClientTimeout(
                                total=timeout * 2,  # Allow 2x for total operation
                                connect=timeout,
                                sock_connect=timeout,
                                sock_read=timeout,
                            )
                            self.session = aiohttp.ClientSession(timeout=client_timeout)

                    return await _orig_method(self, *args, **kwargs)

                # run_coroutine_threadsafe is safe from any thread (Flask, Gevent, gthread)
                return _wrap_future(asyncio.run_coroutine_threadsafe(_coro(), bg_loop))

            setattr(AsyncHTTPClient, method_name, fixed_method)

        # Patch all entry points that might be called from sync code
        patch_method("bulk_upload_custom_metrics")
        patch_method("predict")
        patch_method("async_report_event")

        # Patch close() separately - it's simpler and doesn't need session recreation logic
        def patch_close_method():
            if not hasattr(AsyncHTTPClient, "close"):
                return

            _orig_close = AsyncHTTPClient.close

            def fixed_close(self):
                """
                Thread-safe wrapper for close() that offloads to the background loop.
                Ensures the aiohttp session is properly closed to avoid 'Unclosed client session' warnings.
                """

                async def _close_coro():
                    return await _orig_close(self)

                # Offload to background loop and wait for completion
                return _wrap_future(asyncio.run_coroutine_threadsafe(_close_coro(), bg_loop))

            AsyncHTTPClient.close = fixed_close

        patch_close_method()

    def _patch_otel_utils(self, bg_loop, _wrap_future):
        """Patch OpenTelemetry instrumentation to run async tasks in the background loop."""
        import asyncio

        try:
            import opentelemetry.instrumentation.openai.utils as otel_utils
        except ImportError:
            return

        def handle_task_result(future):
            """Callback to log exceptions from OTEL async tasks."""
            try:
                future.result()
            except Exception:
                logger.exception("OpenTelemetry async task error")

        def fixed_run_async(method):
            """
            Thread-safe wrapper for OTEL async tasks using the shared background loop.
            """
            # Schedule the coroutine safely in the background loop
            fut = asyncio.run_coroutine_threadsafe(method, bg_loop)
            # Add callback to log any exceptions (prevents silent failures)
            fut.add_done_callback(handle_task_result)
            return _wrap_future(fut)

        otel_utils.run_async = fixed_run_async

    def _patch_guard_executor(self, bg_loop, _wrap_future):
        """Patch GuardExecutor to run async_guard_executor in the background loop."""
        import asyncio

        try:
            from datarobot_dome.guard_executor import GuardExecutor
        except ImportError:
            return

        _orig_async_guard_executor = GuardExecutor.async_guard_executor

        def fixed_async_guard_executor(self, *args, **kwargs):
            """
            Thread-safe wrapper for GuardExecutor async tasks using the shared background loop.
            """
            return _wrap_future(
                asyncio.run_coroutine_threadsafe(
                    _orig_async_guard_executor(self, *args, **kwargs), bg_loop
                )
            )

        GuardExecutor.async_guard_executor = fixed_async_guard_executor

    def start(self):
        """
        Starts background tasks for the worker context.

        This method sets the running flag to True and initializes the main application logic.
        It imports and calls the `main` function from `datarobot_drum.drum.main`, passing
        the application instance and the worker context as arguments.
        """
        self._running = True

        self._setup_async_patches()

        from datarobot_drum.drum.main import main

        main(self.app, self)

    def stop(self):
        """
        Gracefully stops the worker context.

        This method sets the running flag to False and executes all registered
        stop callbacks in the order of their priority. Threads and greenlets
        are then joined or killed as part of the stopping process.
        """
        self._running = False
        for _, fn, desc in sorted(self._on_stop, key=lambda x: x[0]):
            try:
                fn()
            except Exception:
                pass

    def cleanup(self):
        """
        Cleans up resources in the worker context.

        This method is called at the end of the worker's lifecycle to release resources.
        It executes all registered cleanup callbacks in reverse priority order, ensuring
        that resources are closed in the correct sequence.

        Exceptions during cleanup are caught and ignored to prevent interruption.
        """
        for _, fn, desc in sorted(self._on_cleanup, key=lambda x: x[0], reverse=True):
            try:
                logger.info("WorkerCtx cleanup: %s", desc)
                fn()
            except Exception as e:
                logger.error("Cleanup failed for '%s': %s", desc, e)

    def running(self) -> bool:
        return self._running


def create_ctx(app):
    """
    Factory method to create a WorkerCtx instance.

    This method initializes the worker context for the application and allows
    registration of objects or callbacks for stopping and cleanup. It ensures
    that no long-running processes are started here; the actual startup occurs
    in `post_worker_init` via `ctx.start()`.

    Args:
        app: The application instance.

    Returns:
        WorkerCtx: The initialized worker context.
    """
    ctx = WorkerCtx(app)
    return ctx
