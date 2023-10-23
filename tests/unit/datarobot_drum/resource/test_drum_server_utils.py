#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import pytest

from datarobot_drum.drum.enum import TargetType
from datarobot_drum.resource.drum_server_utils import DrumServerRun


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

    # @pytest.mark.parametrize("target_type", [el for el in TargetType if el not in TargetType.])
    # def test_other_target_types_ignore_lables(self, target_type):
