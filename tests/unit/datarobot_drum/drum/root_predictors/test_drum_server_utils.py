#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import io
import os
import subprocess
import threading
import time
from queue import Queue
from typing import Tuple, Any
from unittest.mock import patch, Mock

import pytest

from datarobot_drum.drum.enum import TargetType, ArgumentsOptions
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun, DrumServerProcess

from custom_model_runner.datarobot_drum.drum.root_predictors.utils import (
    _queue_output,
    _stream_p_open,
)


@pytest.fixture
def module_under_test():
    return "datarobot_drum.drum.root_predictors.drum_server_utils"


class TestDrumServerRunGetCommand:
    def test_defaults(self):
        target_type = TargetType.BINARY.value
        labels = None
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level=warning "
            f"--code-dir {custom_model_dir} --target-type {target_type} "
            f"--address {runner.server_address} --show-stacktrace --verbose"
        )

        assert runner.get_command() == expected

    def test_with_labels_and_binary(self):
        target_type = TargetType.BINARY.value
        negative_class_label = "nope"
        positive_class_label = "ok-fine!"
        labels = [negative_class_label, positive_class_label]
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level=warning --code-dir "
            f"{custom_model_dir} --target-type {target_type} --address {runner.server_address} "
            f"--positive-class-label '{positive_class_label}' --negative-class-label "
            f"'{negative_class_label}' --show-stacktrace --verbose"
        )

        assert runner.get_command() == expected

    def test_with_labels_and_multiclass(self):
        target_type = TargetType.MULTICLASS.value
        labels = ["a", "b", "c", "d"]
        expected_labels = " ".join([f'"{el}"' for el in labels])
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level=warning --code-dir "
            f"{custom_model_dir} --target-type {target_type} --address {runner.server_address} "
            f"--class-labels {expected_labels} --show-stacktrace --verbose"
        )

        assert runner.get_command() == expected

    @pytest.mark.parametrize(
        "target_type", [el.value for el in TargetType if not el.is_classification()]
    )
    def test_other_target_types_ignore_labels(self, target_type):
        labels = ["a", "b"]
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level=warning --code-dir "
            f"{custom_model_dir} --target-type {target_type} --address {runner.server_address} "
            "--show-stacktrace --verbose"
        )

        assert runner.get_command() == expected

    def test_user_secrets_mount_path(self):
        target_type = TargetType.BINARY.value
        labels = None
        custom_model_dir = "/a/custom/model/dir"
        user_secrets_mount_path = "a/b/c/"
        runner = DrumServerRun(
            target_type, labels, custom_model_dir, user_secrets_mount_path=user_secrets_mount_path
        )

        expected = (
            f"{ArgumentsOptions.MAIN_COMMAND} server --logging-level=warning --code-dir "
            f"{custom_model_dir} --target-type {target_type} --address {runner.server_address} "
            f"--show-stacktrace --verbose --user-secrets-mount-path {user_secrets_mount_path}"
        )

        assert runner.get_command() == expected


@pytest.fixture
def mock_wait_for_server(module_under_test):
    with patch(f"{module_under_test}.wait_for_server") as mock_func:
        yield mock_func


class TestingThread:
    def __init__(self, name, target, args: Tuple[str, DrumServerProcess, Any]):
        self.name = name
        self.target = target
        self.command, self.process_object_holder, self.verbose, self.stream_output = args

    def start(self):
        self.process_object_holder.process = Mock(pid=123)

    def join(self, *args, **kwargs):
        pass

    def is_alive(self):
        return False


@pytest.mark.usefixtures("mock_wait_for_server")
class TestEnter:
    @pytest.fixture
    def runner(self):
        target_type = TargetType.BINARY.value
        labels = None
        custom_model_dir = "/a/custom/model/dir"
        return DrumServerRun(target_type, labels, custom_model_dir, thread_class=TestingThread)

    @pytest.fixture
    def mock_get_command(self):
        with patch.object(DrumServerRun, "get_command") as mock_func:
            mock_func.return_value = "Zhuli, do the thing!"
            yield mock_func

    def test_calls_thread_correctly(self, mock_get_command, runner):
        with runner:
            pass
        assert runner.server_thread.command == mock_get_command.return_value


class TestStreamingOutput:
    def test_queue_output_handles_stdout_and_stderr(self):
        # Create mock objects for stdout, stderr and queue
        mock_stdout = Mock()
        mock_stderr = Mock()
        test_queue = Queue()

        # Configure stdout mock to return a sequence of lines and then stop
        stdout_lines = [b"stdout line 1\n", b"stdout line 2\n", b""]
        mock_stdout.readline.side_effect = stdout_lines

        # Configure stderr mock to return a sequence of lines and then stop
        stderr_lines = [b"stderr line 1\n", b"stderr line 2\n", b""]
        mock_stderr.readline.side_effect = stderr_lines

        # Call the function
        _queue_output(mock_stdout, mock_stderr, test_queue)

        # Verify the queue contains all the expected output lines
        expected_lines = [
            b"stdout line 1\n",
            b"stdout line 2\n",
            b"stderr line 1\n",
            b"stderr line 2\n",
        ]

        # Get all items from the queue and compare with expected
        actual_lines = []
        while not test_queue.empty():
            actual_lines.append(test_queue.get())

        assert actual_lines == expected_lines

        # Verify close was called on both streams
        mock_stdout.close.assert_called_once()
        mock_stderr.close.assert_called_once()

    @patch("builtins.print")
    def test_stream_p_open_handles_process_output(self, mock_print):
        # Create a mock Popen object
        mock_process = Mock(spec=subprocess.Popen)

        # Create real BytesIO objects that will be read by the real _queue_output function
        mock_process.stdout = io.BytesIO(b"stdout line 1\nstdout line 2\n")
        mock_process.stderr = io.BytesIO(b"stderr line 1\nstderr line 2\n")

        # Configure process to terminate after first poll
        mock_process.poll.side_effect = [None, 0]

        # Call the function with real stdout/stderr streams
        stdout, stderr = _stream_p_open(mock_process)

        # Verify the results
        assert stdout == ""
        assert stderr == ""

        # Verify print was called with the expected content
        expected_outputs = [
            b"stdout line 1",
            b"stdout line 2",
            b"stderr line 1",
            b"stderr line 2",
        ]

        # Get the actual arguments passed to print
        actual_outputs = [call_args[0][0] for call_args in mock_print.call_args_list]

        # Verify all expected outputs were printed
        assert len(actual_outputs) == len(expected_outputs)
        for expected in expected_outputs:
            assert expected in actual_outputs

        # Verify poll was called to check for process termination
        assert mock_process.poll.call_count == 2

    @patch("builtins.print")
    def test_stream_p_open_handles_empty_streams(self, mock_print):
        # Create a mock Popen object with empty streams
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdout = io.BytesIO(b"")
        mock_process.stderr = io.BytesIO(b"")

        # Configure process to terminate immediately
        mock_process.poll.return_value = 0

        # Call the function
        stdout, stderr = _stream_p_open(mock_process)

        # Verify results
        assert stdout == ""
        assert stderr == ""
        mock_print.assert_not_called()  # No output to print
        mock_process.poll.assert_called_once()  # Process termination was checked

    @patch("builtins.print")
    def test_stream_p_open_handles_delayed_output(self, mock_print):
        # Create a mock process
        mock_process = Mock(spec=subprocess.Popen)

        # Create pipes that we can write to after the function starts
        r_stdout, w_stdout = os.pipe()
        r_stderr, w_stderr = os.pipe()

        # Convert file descriptors to file-like objects
        mock_process.stdout = os.fdopen(r_stdout, "rb")
        mock_process.stderr = os.fdopen(r_stderr, "rb")

        # Set up process to run and then terminate
        mock_process.poll.side_effect = [None, None, 0]

        # Create a thread to run the function we're testing
        def run_stream_p_open():
            return _stream_p_open(mock_process)

        thread = threading.Thread(target=run_stream_p_open)
        thread.daemon = True
        thread.start()

        # Write data to the pipes
        time.sleep(0.1)  # Give the function time to start
        os.write(w_stdout, b"delayed stdout\n")
        os.write(w_stderr, b"delayed stderr\n")
        os.close(w_stdout)
        os.close(w_stderr)

        # Wait for the thread to finish
        thread.join(timeout=2)
        assert not thread.is_alive()

        # Verify print was called with the expected content
        mock_print.assert_any_call(b"delayed stdout")
        mock_print.assert_any_call(b"delayed stderr")

    @patch("builtins.print")
    def test_stream_p_open_ignores_empty_lines(self, mock_print):
        # Create a mock Popen object
        mock_process = Mock(spec=subprocess.Popen)

        # Create streams with empty lines
        mock_process.stdout = io.BytesIO(b"regular line\n   \n\n")
        mock_process.stderr = io.BytesIO(b"error line\n  \n")

        # Configure process to terminate after output is processed
        mock_process.poll.side_effect = [None, 0]

        # Call the function
        stdout, stderr = _stream_p_open(mock_process)

        # Verify results
        assert stdout == ""
        assert stderr == ""

        # Verify only non-empty lines were printed
        mock_print.assert_any_call(b"regular line")
        mock_print.assert_any_call(b"error line")
        assert mock_print.call_count == 2  # Only two lines should be printed
