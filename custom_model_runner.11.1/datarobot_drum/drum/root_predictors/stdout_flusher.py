import sys
import threading
import time


class StdoutFlusher:
    """An implementation to flush the stdout after a certain time of no activity."""

    def __init__(self, max_time_until_flushing=1.0):
        self._max_time_until_flushing = max_time_until_flushing
        self._last_predict_time = None
        self._flusher_thread = threading.Thread(target=self._flush_thread_method)
        self._flusher_thread.setDaemon(True)
        self._event = threading.Event()
        self._event.clear()

    def start(self):
        """Start the stdout flusher thread."""

        self._flusher_thread.start()

    def is_alive(self):
        """Check if the stdout flusher thread is alive."""

        return self._flusher_thread.is_alive()

    def stop(self):
        """Stop the flusher thread in a synchronous fashion."""

        if self.is_alive():
            self._event.set()
            self._flusher_thread.join()

    def set_last_activity_time(self):
        """Set the last activity time that will be used as the reference for time comparison."""

        self._last_predict_time = self._current_time()

    @staticmethod
    def _current_time():
        return time.time()

    def _flush_thread_method(self):
        while not self._should_stop():
            self._process_stdout_flushing()

    def _should_stop(self):
        return self._event.wait(self._max_time_until_flushing)

    def _process_stdout_flushing(self):
        if self._is_predict_time_set_and_max_waiting_time_expired():
            self._last_predict_time = None
            self._flush_stdout()

    def _is_predict_time_set_and_max_waiting_time_expired(self):
        current_time = self._current_time()
        return (
            self._last_predict_time
            and current_time - self._last_predict_time > self._max_time_until_flushing
        )

    @staticmethod
    def _flush_stdout():
        sys.stdout.flush()
