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

    def start(self):
        """
        Starts background tasks for the worker context.

        This method sets the running flag to True and initializes the main application logic.
        It imports and calls the `main` function from `datarobot_drum.drum.main`, passing
        the application instance and the worker context as arguments.
        """
        self._running = True

        # fix RuntimeError: asyncio.run() cannot be called from a running event loop
        import asyncio

        try:
            import opentelemetry.instrumentation.openai.utils as otel_utils
        except ImportError:
            otel_utils = None

        if otel_utils:

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
                            # Log or handle exceptions here if needed
                            logger.error("OpenTelemetry async task error")

                    # Schedule the coroutine safely and add done callback to handle errors
                    task = loop.create_task(method)
                    task.add_done_callback(handle_task_result)
                else:
                    asyncio.run(method)

            # Apply monkey patch
            otel_utils.run_async = fixed_run_async

        from datarobot_drum.drum.main import main
        import os
        os.chdir(os.environ.get("CODE_DIR", "/opt/code"))

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
                logger.error("Tracing shutdown failed: %s", e)

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
