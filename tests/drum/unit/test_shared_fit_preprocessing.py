"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import tempfile

from datarobot_drum.drum.utils import shared_fit_preprocessing

import numpy as np

import pytest


@pytest.fixture
def mock_fit_class():
    """
    Dummy fit class object intended to be modified for specific test cases.
    """

    class MockFitClass(object):
        num_rows = "ALL"
        input_filename = None

        target_filename = None
        target_name = None

        parameter_file = None
        default_parameter_values = None

        weights_filename = None
        weights = None

        class_labels = None
        negative_class_label = None
        positive_class_label = None

    return MockFitClass()


def test_single_col_of_data_treats_missing_lines_as_nans(mock_fit_class):
    """
    Shared fit preprocessing should interpret a blank row as nan if the CSV contains a single column
    """
    # row 5 should be nan, and should not count the last line as a nan (totalling 7 rows)
    single_col_csv_data = "data\n0\n1\n2\n3\n4\n\n6\n"

    tmp_file = tempfile.NamedTemporaryFile(suffix=".csv")
    with open(tmp_file.name, "w") as f:
        f.write(single_col_csv_data)

    mock_fit_class.input_filename = tmp_file.name

    X, _, _, _, _ = shared_fit_preprocessing(mock_fit_class)

    tmp_file.close()

    assert np.isnan(X["data"][5])
    assert X.shape[0] == 7
