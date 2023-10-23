import time
from unittest.mock import patch

from datarobot_drum.resource.components.Python.prediction_server.stdout_flusher import StdoutFlusher


class TestStdoutFlusher:
    """Contains cases to test the StdoutFlusher functionality."""

    def test_start_and_stop(self):
        """Test the life-cycle methods and validate liveliness."""

        stdout_flusher = StdoutFlusher()
        assert not stdout_flusher.is_alive()
        stdout_flusher.stop()
        assert not stdout_flusher.is_alive()
        stdout_flusher.start()
        try:
            assert stdout_flusher.is_alive()
        finally:
            stdout_flusher.stop()
            assert not stdout_flusher.is_alive()

    def test_no_prediction_activity(self):
        """
        Test a scenario in which predictions were not submitted and therefore no need to flush
        stdout.
        """

        max_time_until_flushing = 0.01
        with patch.object(
            StdoutFlusher, "_is_predict_time_set_and_max_waiting_time_expired", return_value=False
        ) as condition_method, patch.object(StdoutFlusher, "_flush_stdout") as flush_stdout:
            stdout_flusher = StdoutFlusher(max_time_until_flushing=max_time_until_flushing)
            stdout_flusher.start()
            try:
                time.sleep(max_time_until_flushing * 2)
                condition_method.assert_called()
                flush_stdout.assert_not_called()
            finally:
                stdout_flusher.stop()

    def test_no_required_stdout_flush(self):
        """
        Test a scenario in which the max wait time was not expired yet and therefore no need to
        flush stdout.
        """

        max_time_until_flushing = 0.1
        last_predict_time = 10
        current_time = last_predict_time + max_time_until_flushing / 2
        self._test_stdout_flush(
            max_time_until_flushing, last_predict_time, current_time, flushing_expected=False
        )

    def test_required_stdout_flush(self):
        """
        Test a scenario in which the max wait time was exceeded, and therefore it is required to
        flush the stdout.
        """

        max_time_until_flushing = 0.1
        last_predict_time = 10
        current_time = last_predict_time + max_time_until_flushing * 2
        self._test_stdout_flush(
            max_time_until_flushing, last_predict_time, current_time, flushing_expected=True
        )

    @staticmethod
    def _test_stdout_flush(max_wait_time, last_predict_time, current_time, flushing_expected):
        with patch.object(
            StdoutFlusher, "_current_time", side_effect=[last_predict_time, current_time]
        ), patch.object(StdoutFlusher, "_flush_stdout") as flush_stdout:
            stdout_flusher = StdoutFlusher(max_time_until_flushing=max_wait_time)
            stdout_flusher.set_last_activity_time()
            stdout_flusher.start()
            try:
                time.sleep(max_wait_time * 2)
                if flushing_expected:
                    flush_stdout.assert_called()
                else:
                    flush_stdout.assert_not_called()
            finally:
                stdout_flusher.stop()
