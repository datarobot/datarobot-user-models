import sys
import time
try:
    import gevent
    from gevent import Greenlet
    HAS_GEVENT = True
except ImportError:
    HAS_GEVENT = False
    import threading

HAS_GEVENT = True


class GeventCompatibleStdoutFlusher:
    """An implementation to flush the stdout after a certain time of no activity.
    Compatible with both gevent and threading environments."""

    def __init__(self, max_time_until_flushing=1.0):
        self._max_time_until_flushing = max_time_until_flushing
        self._last_predict_time = None
        self._flusher_greenlet = None
        self._flusher_thread = None
        self._stop_event = None
        self._running = False

    def start(self):
        """Start the stdout flusher."""
        if self._running:
            return

        self._running = True

        if HAS_GEVENT:
            self._flusher_greenlet = gevent.spawn(self._flush_greenlet_method)
        else:
            self._stop_event = threading.Event()
            self._flusher_thread = threading.Thread(target=self._flush_thread_method)
        """Check if the stdout flusher is alive."""
        if HAS_GEVENT and self._flusher_greenlet:
            return not self._flusher_greenlet.dead
        elif self._flusher_thread:
            return self._flusher_thread.is_alive()
        return False

    def stop(self):
        """Stop the flusher in a synchronous fashion."""
        if not self._running:
            return

        self._running = False

        if HAS_GEVENT and self._flusher_greenlet:
            self._flusher_greenlet.kill()
            self._flusher_greenlet = None
        elif self._flusher_thread and self._stop_event:
            self._stop_event.set()
            self._flusher_thread.join(timeout=2.0)  # Timeout to prevent hanging
            self._flusher_thread = None
            self._stop_event = None

    def set_last_activity_time(self):
        """Set the last activity time that will be used as the reference for time comparison."""
        self._last_predict_time = self._current_time()

    @staticmethod
    def _current_time():
        return time.time()

    def _flush_greenlet_method(self):
        """Gevent greenlet method for stdout flushing"""
        try:
            while self._running:
                self._process_stdout_flushing()
                #gevent.sleep(self._max_time_until_flushing)
                gevent.sleep(0)
        except gevent.GreenletExit:
            pass  # Normal termination

    def _flush_thread_method(self):
        """Threading method for stdout flushing"""
        while self._running and not self._stop_event.wait(self._max_time_until_flushing):
            self._process_stdout_flushing()

    def _process_stdout_flushing(self):
        if self._is_predict_time_set_and_max_waiting_time_expired():
            sys.stdout.flush()
            sys.stderr.flush()

    def _is_predict_time_set_and_max_waiting_time_expired(self):
        if self._last_predict_time is None:
            return False

        return (self._current_time() - self._last_predict_time) >= self._max_time_until_flushing
