import logging
import threading
from typing import Callable, Any, List, Tuple, Optional

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def _is_gevent_patched() -> bool:
    """Check if gevent monkey patching is active for threading module."""
    try:
        from gevent import monkey

        return monkey.is_module_patched("threading")
    except ImportError:
        return False


def _get_real_threading_lock():
    """
    Get a real threading.Lock that is NOT patched by gevent.

    When gevent monkey patches threading, threading.Lock becomes a gevent lock
    which can cause deadlocks when used across real OS threads (like the asyncio
    background event loop thread).

    Returns:
        A real threading.Lock if gevent is patched, otherwise a regular threading.Lock.
    """
    if _is_gevent_patched():
        try:
            from gevent import monkey

            # Get the original unpatched threading module
            _real_threading = monkey.get_original("threading", "Lock")
            if _real_threading:
                return _real_threading()
        except (ImportError, AttributeError):
            pass
    return threading.Lock()


def _get_real_threading_thread():
    """
    Get a real threading.Thread class that is NOT patched by gevent.

    When gevent monkey patches threading, threading.Thread creates greenlets instead
    of real OS threads. For background event loops that need true thread isolation
    (like asyncio loops used with run_coroutine_threadsafe), we need real OS threads.

    Returns:
        The real threading.Thread class if gevent is patched, otherwise the regular class.
    """
    if _is_gevent_patched():
        try:
            from gevent import monkey

            RealThread = monkey.get_original("threading", "Thread")
            if RealThread:
                return RealThread
        except (ImportError, AttributeError):
            pass
    return threading.Thread


def _wait_for_future_gevent_safe(fut, timeout=None):
    """
    Wait for a concurrent.futures.Future in a gevent-safe manner.

    Blocking calls like fut.result() can block the entire gevent hub when called
    from a greenlet context. This function polls the future with cooperative yields.

    Args:
        fut: A concurrent.futures.Future to wait for.
        timeout: Maximum time to wait in seconds (None = wait forever).

    Returns:
        The result of the future.

    Raises:
        TimeoutError: If timeout is exceeded.
        Any exception raised by the future.
    """
    if not _is_gevent_patched():
        # Not in gevent context, use regular blocking wait
        return fut.result(timeout=timeout)

    import time

    try:
        import gevent
    except ImportError:
        return fut.result(timeout=timeout)

    start_time = time.monotonic()
    poll_interval = 0.01  # Start with 10ms polling

    while not fut.done():
        if timeout is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Future did not complete within {timeout} seconds")

        # Cooperative yield to allow other greenlets to run
        gevent.sleep(poll_interval)

        # Exponential backoff up to 100ms to reduce CPU usage for long operations
        poll_interval = min(poll_interval * 1.5, 0.1)

    return fut.result()  # Will raise if future had an exception


def _join_thread_gevent_safe(thread, timeout):
    """
    Join a thread with gevent-safe waiting if needed.

    When gevent monkey patches threading, blocking on thread.join() from a greenlet
    context will block the entire gevent hub. This function uses cooperative polling
    to allow other greenlets to run while waiting for the thread to terminate.

    Args:
        thread: The thread to join.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if the thread terminated, False if timeout was reached.
    """
    if not thread.is_alive():
        return True

    if _is_gevent_patched():
        import time

        try:
            import gevent
        except ImportError:
            thread.join(timeout=timeout)
            return not thread.is_alive()

        start = time.monotonic()
        poll_interval = 0.05  # Start with 50ms polling

        while thread.is_alive():
            if time.monotonic() - start >= timeout:
                return False
            gevent.sleep(poll_interval)
            # Exponential backoff up to 200ms to reduce CPU usage
            poll_interval = min(poll_interval * 1.5, 0.2)
        return True
    else:
        thread.join(timeout=timeout)
        return not thread.is_alive()


class _GeventFutureWrapper:
    """
    Wrapper for concurrent.futures.Future that provides gevent-safe waiting.

    This wrapper intercepts .result() calls and uses cooperative polling
    instead of blocking, allowing other greenlets to run while waiting.
    """

    def __init__(self, fut):
        self._fut = fut

    def result(self, timeout=None):
        """Get the result with gevent-safe cooperative waiting."""
        return _wait_for_future_gevent_safe(self._fut, timeout=timeout)

    def done(self):
        """Check if the future is done."""
        return self._fut.done()

    def cancelled(self):
        """Check if the future was cancelled."""
        return self._fut.cancelled()

    def add_done_callback(self, fn):
        """Add a callback to be called when the future completes."""
        return self._fut.add_done_callback(fn)

    def exception(self, timeout=None):
        """Get the exception with gevent-safe cooperative waiting.

        Unlike result(), this method returns the exception instead of raising it.
        Uses cooperative polling to wait for future completion without blocking
        the gevent hub.
        """
        if not _is_gevent_patched():
            # Not in gevent context, use regular blocking wait
            return self._fut.exception(timeout=timeout)

        import time

        try:
            import gevent
        except ImportError:
            return self._fut.exception(timeout=timeout)

        start_time = time.monotonic()
        poll_interval = 0.01  # Start with 10ms polling

        while not self._fut.done():
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Future did not complete within {timeout} seconds")

            # Cooperative yield to allow other greenlets to run
            gevent.sleep(poll_interval)

            # Exponential backoff up to 100ms to reduce CPU usage for long operations
            poll_interval = min(poll_interval * 1.5, 0.1)

        # Future is done, get the exception (returns None if no exception)
        return self._fut.exception(timeout=0)

    def __getattr__(self, name):
        """Delegate other attributes to the underlying future."""
        return getattr(self._fut, name)


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
                # Use gevent-safe joining to avoid blocking the hub when called from greenlet context
                _join_thread_gevent_safe(t, join_timeout)

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

        Gevent Compatibility:
        - Uses real (unpatched) threading.Lock for cross-thread synchronization
        - Provides gevent-safe future waiting that yields cooperatively
        - Handles cleanup without blocking the gevent hub
        """
        import asyncio

        def _wrap_future(fut):
            """
            Wrap concurrent.futures.Future for the current execution context.

            - If inside bg_loop (async context in real thread): wrap as asyncio.Future (awaitable)
            - If in gevent context without loop: return GeventFutureWrapper for cooperative waiting
            - Otherwise: return the raw future for blocking .result() calls

            The key insight: bg_loop runs in a real OS thread (not greenlet), so
            asyncio.wrap_future is safe there even with gevent active elsewhere.
            """
            # First check if we're in an async context
            try:
                current_loop = asyncio.get_running_loop()
                # We're in some running event loop

                # If it's bg_loop, we're in the real OS thread - safe to use wrap_future
                if current_loop is bg_loop:
                    return asyncio.wrap_future(fut)

                # If it's some other loop and gevent is active, we might be in a greenlet
                # with an event loop, which is problematic for wrap_future in Python 3.12
                if _is_gevent_patched():
                    # Return GeventFutureWrapper for sync-style waiting
                    # Note: This means the caller cannot await this - they must call .result()
                    return _GeventFutureWrapper(fut)

                # Not gevent, some other loop - safe to wrap
                return asyncio.wrap_future(fut)

            except RuntimeError:
                # No running loop - we're in sync context
                pass

            # In gevent sync context (no loop), use cooperative wrapper
            if _is_gevent_patched():
                return _GeventFutureWrapper(fut)

            # Plain sync context
            return fut

        # Background loop for thread-safe async execution in sync/gevent workers.
        # We store it on self.app to ensure it lives as long as the worker.
        # Guard both loop creation AND patching to prevent nested wrappers if start() is called multiple times.
        if hasattr(self.app, "_drum_bg_loop"):
            return  # Already initialized - skip to avoid double-patching

        loop = asyncio.new_event_loop()
        # IMPORTANT: Use real (unpatched) threading.Thread to create a real OS thread.
        # When gevent monkey patches threading, Thread creates greenlets instead of real threads.
        # The asyncio event loop MUST run in a real OS thread for run_coroutine_threadsafe()
        # to work correctly - it relies on thread-safe wakeup mechanisms.
        RealThread = _get_real_threading_thread()
        bg_thread = RealThread(target=loop.run_forever, daemon=True, name="DrumBgLoop")
        bg_thread.start()
        self.app._drum_bg_loop = loop
        self.app._drum_bg_thread = bg_thread  # Store thread reference for cleanup

        bg_loop = self.app._drum_bg_loop

        # Register cleanup for the background loop to allow pending async operations
        # (metrics uploads, guard executions, telemetry) to complete during shutdown.
        def _cleanup_bg_loop():
            import gc
            import time

            # Capture thread reference in closure to avoid variable shadowing issues
            loop_thread = self.app._drum_bg_thread

            # Install a custom exception handler to suppress OSError during shutdown.
            # When gevent patches sockets and asyncio transports get garbage collected,
            # they may try to close sockets that gevent has already detached, causing
            # "Bad file descriptor" errors. These are benign since the worker is exiting.
            def _shutdown_exception_handler(loop, context):
                exc = context.get("exception")
                if isinstance(exc, OSError) and exc.errno == 9:  # EBADF
                    logger.debug("Suppressed OSError during shutdown: %s", exc)
                    return
                # For other exceptions, use default handling
                loop.default_exception_handler(context)

            try:
                bg_loop.set_exception_handler(_shutdown_exception_handler)
            except Exception:
                pass  # Loop might be closed or not support this

            # Check if loop is still running before attempting graceful shutdown
            if not bg_loop.is_running():
                logger.debug("Background loop already stopped, skipping graceful shutdown")
                # Still need to join thread and close loop
                _join_thread_gevent_safe(loop_thread, 2.0)
                # Run GC to trigger transport finalizers while we can still handle errors
                gc.collect()
                try:
                    bg_loop.close()
                except Exception:
                    pass
                return

            # Helper to wait for thread termination without relying on gevent.sleep
            # During shutdown, gevent may not process sleep() correctly
            def _wait_for_thread_stop(thread, timeout):
                """Wait for thread using time.sleep polling (works during gevent shutdown)."""
                start = time.monotonic()
                poll_interval = 0.05
                while thread.is_alive():
                    if time.monotonic() - start >= timeout:
                        return False
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 0.2)
                return True

            graceful_shutdown_succeeded = False
            try:
                # Give pending tasks a chance to complete by scheduling a graceful shutdown
                async def _shutdown():
                    tasks = [
                        task
                        for task in asyncio.all_tasks(bg_loop)
                        if task is not asyncio.current_task()
                    ]
                    if tasks:
                        # Cancel all tasks immediately - don't wait for graceful completion
                        # during worker restart to minimize shutdown time
                        for task in tasks:
                            task.cancel()
                        # Give cancelled tasks a brief moment to clean up
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            logger.debug("Some tasks did not respond to cancellation")

                    # Properly shutdown async generators and default executor
                    try:
                        await bg_loop.shutdown_asyncgens()
                    except Exception as e:
                        logger.debug("Error shutting down async generators: %s", e)

                    try:
                        await bg_loop.shutdown_default_executor()
                    except Exception as e:
                        logger.debug("Error shutting down default executor: %s", e)

                    bg_loop.stop()

                # Schedule the shutdown coroutine
                shutdown_fut = asyncio.run_coroutine_threadsafe(_shutdown(), bg_loop)

                # Wait for completion using time.sleep polling (not gevent.sleep)
                # This is more reliable during gevent shutdown
                start_time = time.monotonic()
                timeout = 3.0  # Shorter timeout for faster worker restarts
                poll_interval = 0.05

                while not shutdown_fut.done():
                    if time.monotonic() - start_time >= timeout:
                        raise TimeoutError("Graceful shutdown timed out")
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 0.1)

                # Check if the future completed with an exception
                shutdown_fut.result(timeout=0)
                graceful_shutdown_succeeded = True

            except (KeyboardInterrupt, SystemExit) as e:
                # Handle interruption signals during shutdown - force stop
                logger.debug("Interrupted during background loop shutdown: %s", type(e).__name__)
            except TimeoutError:
                # Graceful shutdown timed out - this is expected during fast worker restarts
                logger.debug("Graceful background loop shutdown timed out, forcing stop")
            except Exception as e:
                logger.debug("Background loop shutdown error: %s", e)
            finally:
                # Force stop the loop if graceful shutdown didn't complete
                if not graceful_shutdown_succeeded:
                    try:
                        bg_loop.call_soon_threadsafe(bg_loop.stop)
                    except RuntimeError:
                        pass  # Loop may already be stopped/closed

                # Wait for the thread to terminate using time.sleep (not gevent.sleep)
                if not _wait_for_thread_stop(loop_thread, 2.0):
                    logger.debug("Background loop thread did not terminate within timeout")

                # Run garbage collection to finalize any pending transports
                gc.collect()

                # Close the loop, suppressing OSError from gevent/asyncio socket conflicts
                try:
                    bg_loop.close()
                except OSError as e:
                    logger.debug(
                        "OSError during background loop close (expected with gevent): %s", e
                    )
                except Exception as e:
                    logger.debug("Error closing background loop: %s", e)

                # Final GC pass after loop is closed
                gc.collect()

        # Use low order (negative) so loop cleanup happens after other cleanup callbacks
        # that might still need to use async operations
        self.defer_cleanup(_cleanup_bg_loop, order=-100, desc="background event loop")

        self._patch_async_http_client(bg_loop, _wrap_future)
        self._patch_otel_utils(bg_loop, _wrap_future)
        self._patch_guard_executor(bg_loop, _wrap_future)
        self._patch_nemo_rails(bg_loop, _wrap_future)

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
        # Track client instances for session cleanup during shutdown.
        # WeakSet allows automatic removal when clients are garbage collected.
        _tracked_clients: weakref.WeakSet = weakref.WeakSet()

        # asyncio.Lock to protect the check-then-set pattern in _get_session_lock.
        # This protects dictionary access from race conditions when multiple coroutines
        # concurrently check for an existing lock.
        # Using asyncio.Lock instead of threading.Lock because this is used exclusively
        # within async coroutines in bg_loop - asyncio.Lock provides cooperative yielding
        # instead of blocking the event loop.
        # Create the lock in bg_loop context to ensure proper event loop binding.
        async def _create_async_lock():
            return asyncio.Lock()

        fut = asyncio.run_coroutine_threadsafe(_create_async_lock(), bg_loop)
        _session_locks_guard = _wait_for_future_gevent_safe(fut, timeout=5.0)
        # asyncio.Lock to protect _tracked_clients WeakSet from concurrent modification.
        # Using asyncio.Lock instead of threading.Lock because this is used exclusively
        # within async coroutines in bg_loop - asyncio.Lock provides cooperative yielding
        # instead of blocking the event loop.
        fut = asyncio.run_coroutine_threadsafe(_create_async_lock(), bg_loop)
        _tracked_clients_guard = _wait_for_future_gevent_safe(fut, timeout=5.0)

        def _cleanup_all_sessions():
            """Close all tracked aiohttp sessions during shutdown.

            This prevents 'Unclosed client session' warnings from aiohttp.
            Must be called before the background loop is stopped.
            """
            if not bg_loop.is_running():
                logger.debug("Background loop not running, skipping session cleanup")
                return

            async def _close_sessions():
                closed_count = 0
                # Copy the set under lock to avoid concurrent modification
                async with _tracked_clients_guard:
                    clients_snapshot = list(_tracked_clients)
                for client in clients_snapshot:
                    try:
                        session = getattr(client, "session", None)
                        if session and not session.closed:
                            await session.close()
                            closed_count += 1
                    except Exception as e:
                        logger.debug("Error closing session for client %s: %s", id(client), e)
                if closed_count:
                    logger.debug("Closed %d aiohttp session(s) during cleanup", closed_count)

            try:
                fut = asyncio.run_coroutine_threadsafe(_close_sessions(), bg_loop)
                _wait_for_future_gevent_safe(fut, timeout=5.0)
            except Exception as e:
                logger.debug("Error during session cleanup: %s", e)

        # Register session cleanup BEFORE background loop cleanup (higher order = earlier execution)
        # This ensures sessions are closed while the loop is still running
        self.defer_cleanup(_cleanup_all_sessions, order=-50, desc="aiohttp sessions")

        async def _get_session_lock(client) -> asyncio.Lock:
            """Get or create an asyncio.Lock for the given client instance.

            Must be called from within an async context (inside bg_loop).
            Uses asyncio.Lock for cooperative yielding instead of blocking the event loop.
            """
            async with _session_locks_guard:
                if client not in _session_locks:
                    # asyncio.Lock() created inside async context is properly bound to running loop
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
                    logger.debug("Could not determine session's event loop, will recreate session")
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
                    async with await _get_session_lock(self):
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
                            # Track this client for cleanup during shutdown
                            async with _tracked_clients_guard:
                                _tracked_clients.add(self)

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

    def _patch_nemo_rails(self, bg_loop, _wrap_future):
        """Patch LLMRails to run generate_async in the background loop."""
        import asyncio

        try:
            from nemoguardrails import LLMRails
        except ImportError:
            return

        _orig_generate_async = LLMRails.generate_async

        def fixed_generate_async(self, *args, **kwargs):
            """
            Thread-safe wrapper for LLMRails.generate_async that offloads execution
            to the dedicated background event loop.

            This ensures the coroutine runs in bg_loop, following the same pattern as
            other patches (_patch_guard_executor, _patch_async_http_client) which use
            non-async functions returning _wrap_future(asyncio.run_coroutine_threadsafe(...)).

            Returns a wrapped future that can be:
            - Awaited in async contexts (when in bg_loop)
            - Called with .result() in sync/gevent contexts (cooperative waiting)
            """
            return _wrap_future(
                asyncio.run_coroutine_threadsafe(
                    _orig_generate_async(self, *args, **kwargs), bg_loop
                )
            )

        LLMRails.generate_async = fixed_generate_async

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
