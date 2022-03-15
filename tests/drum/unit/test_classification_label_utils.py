import os
from argparse import Namespace
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from datarobot_drum.drum.custom_tasks.fit_adapters.classification_labels_util import (
    possibly_intuit_order,
)
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.enum import TargetType, RunMode
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.runtime import DrumRuntime
from drum.constants import TESTS_DATA_PATH


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


@pytest.mark.parametrize(
    "target_type, expected_pos_label, expected_neg_label, expected_class_labels",
    [
        (TargetType.BINARY, "Iris-setosa", "Iris-versicolor", None),
        (TargetType.MULTICLASS, None, None, ["Iris-setosa", "Iris-versicolor"]),
    ],
)
def test_class_labels_from_target(
    target_type, expected_pos_label, expected_neg_label, expected_class_labels
):
    # TODO: unit test this without namespace
    test_data_path = os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
    with DrumRuntime() as runtime:
        runtime.options = Namespace(
            negative_class_label=None,
            positive_class_label=None,
            class_labels=None,
            code_dir="",
            disable_strict_validation=False,
            logging_level="warning",
            subparser_name=RunMode.FIT,
            target_type=target_type,
            verbose=False,
            content_type=None,
            input=test_data_path,
            target_csv=None,
            target="Species",
            row_weights=None,
            row_weights_csv=None,
            output=None,
            num_rows=0,
            sparse_column_file=None,
            parameter_file=None,
        )
        cmrunner = CMRunner(runtime)
        cmrunner._prepare_fit()

        assert cmrunner.options.negative_class_label == expected_neg_label
        assert cmrunner.options.positive_class_label == expected_pos_label
        assert cmrunner.options.class_labels == expected_class_labels
