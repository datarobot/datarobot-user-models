import os

import pytest

from datarobot_drum.drum.adapters.cli.shared.drum_class_label_adapter import DrumClassLabelAdapter
from datarobot_drum.drum.enum import TargetType
from drum.constants import TESTS_DATA_PATH


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

        drum_cli_adapter = DrumClassLabelAdapter(target_type=target_type,)

        drum_cli_adapter._infer_class_labels_if_not_provided(
            input_filename=test_data_path, target_name=test_target_name,
        )
        assert drum_cli_adapter.negative_class_label == expected_neg_label
        assert drum_cli_adapter.positive_class_label == expected_pos_label
        assert drum_cli_adapter.class_labels == expected_class_labels
        assert drum_cli_adapter.class_ordering == expected_class_ordering
