#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from subprocess import Popen
from typing import Tuple, Any
from unittest.mock import patch, Mock

import pytest

from datarobot_drum.drum.enum import TargetType
from datarobot_drum.resource.drum_server_utils import DrumServerRun, DrumServerProcess


@pytest.fixture
def module_under_test():
    return "datarobot_drum.resource.drum_server_utils"


class TestDrumServerRunGetCommand:
    def test_defaults(self):
        target_type = TargetType.BINARY.value
        labels = None
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"drum server --logging-level=warning --code-dir {custom_model_dir} --target-type {target_type} "
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
            f"drum server --logging-level=warning --code-dir {custom_model_dir} --target-type {target_type} "
            f"--address {runner.server_address} --positive-class-label '{positive_class_label}' "
            f"--negative-class-label '{negative_class_label}' --show-stacktrace --verbose"
        )

        assert runner.get_command() == expected

    def test_with_labels_and_multiclass(self):
        target_type = TargetType.MULTICLASS.value
        labels = ["a", "b", "c", "d"]
        expected_labels = " ".join([f'"{el}"' for el in labels])
        custom_model_dir = "/a/custom/model/dir"
        runner = DrumServerRun(target_type, labels, custom_model_dir)

        expected = (
            f"drum server --logging-level=warning --code-dir {custom_model_dir} --target-type {target_type} "
            f"--address {runner.server_address} --class-labels {expected_labels} "
            f"--show-stacktrace --verbose"
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
            f"drum server --logging-level=warning --code-dir {custom_model_dir} --target-type {target_type} "
            f"--address {runner.server_address} --show-stacktrace --verbose"
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
            f"drum server --logging-level=warning --code-dir {custom_model_dir} --target-type {target_type} "
            f"--address {runner.server_address} --show-stacktrace --verbose "
            f"--user-secrets-mount-path {user_secrets_mount_path}"
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
        self.command, self.process_object_holder, self.verbose = args

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
