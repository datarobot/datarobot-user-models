#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

import os
from tempfile import NamedTemporaryFile

import pytest
import pandas as pd
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import possibly_intuit_order
from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.exceptions import DrumCommonException


from tests.constants import TESTS_DATA_PATH


class TestDrumCLIAdapterLabels(object):
    @pytest.mark.parametrize(
        "target_type, expected_pos_label, expected_neg_label, expected_class_labels, expected_class_ordering",
        [
            (
                TargetType.BINARY,
                "Iris-setosa",
                "Iris-versicolor",
                None,
                ["Iris-setosa", "Iris-versicolor"],
            ),
            (
                TargetType.MULTICLASS,
                None,
                None,
                ["Iris-setosa", "Iris-versicolor"],
                ["Iris-setosa", "Iris-versicolor"],
            ),
            (TargetType.REGRESSION, None, None, None, None),
            (TargetType.ANOMALY, None, None, None, None),
            (TargetType.TRANSFORM, None, None, None, None),
            (TargetType.UNSTRUCTURED, None, None, None, None),
            (TargetType.TEXT_GENERATION, None, None, None, None),
        ],
    )
    def test_infer_class_labels_if_not_provided(
        self,
        target_type,
        expected_pos_label,
        expected_neg_label,
        expected_class_labels,
        expected_class_ordering,
    ):
        test_data_path = os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
        test_target_name = "Species"

        drum_cli_adapter = DrumClassLabelAdapter(
            target_type=target_type,
        )

        drum_cli_adapter._infer_class_labels_if_not_provided(
            input_filename=test_data_path,
            target_name=test_target_name,
        )
        assert drum_cli_adapter.negative_class_label == expected_neg_label
        assert drum_cli_adapter.positive_class_label == expected_pos_label
        assert drum_cli_adapter.class_labels == expected_class_labels
        assert drum_cli_adapter.class_ordering == expected_class_ordering


class TestOrderIntuition:
    binary_filename = os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
    regression_filename = os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression.csv")
    one_target_filename = os.path.join(TESTS_DATA_PATH, "one_target.csv")

    def test_colname(self):
        classes = possibly_intuit_order(
            input_filename=self.binary_filename,
            target_type=TargetType.BINARY,
            target_name="Species",
        )
        assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_colfile(self):
        with NamedTemporaryFile() as target_file:
            df = pd.read_csv(self.binary_filename)
            with open(target_file.name, "w") as f:
                target_series = df["Species"]
                target_series.to_csv(f, index=False, header="Target")

            classes = possibly_intuit_order(
                input_filename=self.binary_filename,
                target_type=TargetType.BINARY,
                target_filename=target_file.name,
            )
            assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_badfile(self):
        with pytest.raises(DrumCommonException):
            possibly_intuit_order(
                input_filename=self.one_target_filename,
                target_type=TargetType.BINARY,
                target_name="Species",
            )

    def test_unsupervised(self):
        classes = possibly_intuit_order(
            input_filename=self.regression_filename,
            target_type=TargetType.ANOMALY,
            target_name="Grade 2014",
        )
        assert classes is None
