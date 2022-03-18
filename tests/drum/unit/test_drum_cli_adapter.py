"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os

import pytest
import pandas as pd
import numpy as np
import scipy
import scipy.sparse

from datarobot_drum.drum.adapters.drum_cli_adapter import DrumCLIAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe
from drum.constants import TESTS_DATA_PATH


@pytest.fixture
def num_rows():
    return 10


@pytest.fixture
def num_cols():
    return 5


@pytest.fixture
def column_names(num_cols):
    return ["c" + str(i) for i in range(num_cols)]


@pytest.fixture
def target_col_name():
    return "target_col"


@pytest.fixture
def weights_col_name():
    return "weights_col_name"


@pytest.fixture
def col_data(num_rows):
    return np.array(range(num_rows))  # each col has values 0, 1, 2, ..., (num_rows - 1)


@pytest.fixture
def target_data_offset():
    return 0.1


@pytest.fixture
def target_series(col_data, target_data_offset, target_col_name):
    # offset target data so its not equal to X's col values
    return pd.Series(col_data + target_data_offset, name=target_col_name)


@pytest.fixture
def weights_data_offset():
    return 0.2


@pytest.fixture
def weights_series(col_data, weights_data_offset, weights_col_name):
    # offset weights data so its not equal to X's or target's col values
    return pd.Series(col_data + weights_data_offset, name=weights_col_name)


@pytest.fixture
def dense_df(col_data, num_cols, column_names):
    data = np.repeat([[x] for x in col_data], num_cols, axis=1)
    return pd.DataFrame(data, columns=column_names)


@pytest.fixture
def dense_df_with_target(dense_df, target_series):
    return pd.concat([dense_df, target_series], axis=1)


@pytest.fixture
def dense_df_with_target_and_weights(dense_df, target_series, weights_series):
    return pd.concat([dense_df, target_series, weights_series], axis=1)


@pytest.fixture
def sparse_df(dense_df, column_names):
    data = dense_df.values
    sparse_data = scipy.sparse.coo_matrix(data)

    sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_data)
    sparse_df.columns = column_names
    return sparse_df


@pytest.fixture
def dense_csv(dense_df, df_to_temporary_file):
    with df_to_temporary_file(dense_df) as filename:
        yield filename


@pytest.fixture
def dense_csv_with_target(dense_df_with_target, df_to_temporary_file):
    with df_to_temporary_file(dense_df_with_target) as filename:
        yield filename


@pytest.fixture
def dense_csv_with_target_and_weights(dense_df_with_target_and_weights, df_to_temporary_file):
    with df_to_temporary_file(dense_df_with_target_and_weights) as filename:
        yield filename


@pytest.fixture
def sparse_mtx(sparse_df, df_to_temporary_file):
    with df_to_temporary_file(sparse_df) as filename:
        yield filename


@pytest.fixture
def target_csv(target_series, df_to_temporary_file):
    with df_to_temporary_file(target_series) as filename:
        yield filename


@pytest.fixture
def weights_csv(weights_series, df_to_temporary_file):
    with df_to_temporary_file(weights_series) as filename:
        yield filename


@pytest.fixture
def sparse_column_names_file(column_names, df_to_temporary_file):
    with df_to_temporary_file(pd.DataFrame(column_names), header=False) as filename:
        yield filename


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

        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="arbitrary/path/to/nowhere",
            input_filename=test_data_path,
            target_type=target_type,
            target_name="Species",
        )

        drum_cli_adapter._infer_class_labels_if_not_provided()
        assert drum_cli_adapter.negative_class_label == expected_neg_label
        assert drum_cli_adapter.positive_class_label == expected_pos_label
        assert drum_cli_adapter.class_labels == expected_class_labels
        assert drum_cli_adapter.class_ordering == expected_class_ordering


class TestDrumCLIAdapterFailures(object):
    @pytest.mark.parametrize(
        "target_type",
        [TargetType.BINARY, TargetType.REGRESSION, TargetType.MULTICLASS, TargetType.TRANSFORM],
    )
    def test_target_data_missing(self, dense_csv, target_type):
        with pytest.raises(
            DrumCommonException, match="Must provide target name or target filename to drum fit"
        ):
            _ = DrumCLIAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=target_type,
            ).y

    def test_input_file_contains_X_missing_y(
        self, dense_csv, target_col_name,
    ):
        with pytest.raises(
            DrumCommonException,
            match=f"The target column '{target_col_name}' does not exist in your input data.",
        ):
            _ = DrumCLIAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=TargetType.REGRESSION,
                target_name=target_col_name,
            ).y

    def test_input_file_contains_X_missing_weights(
        self, dense_csv, weights_col_name,
    ):
        with pytest.raises(
            DrumCommonException,
            match=f"The weights column '{weights_col_name}' does not exist in your input data.",
        ):
            _ = DrumCLIAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=TargetType.REGRESSION,
                weights_name=weights_col_name,
            ).weights

    def test_output_dir_cannot_equal_custom_task_folder_path(self):
        with pytest.raises(
            DrumCommonException, match="The code directory may not be used as the output directory."
        ):
            _ = DrumCLIAdapter(
                custom_task_folder_path="conflicting/path",
                input_filename="path/to/nothing",
                target_type=TargetType.REGRESSION,
                output_dir="conflicting/path",
            )._validate_output_dir()

    def test_num_rows_greater_than_input_data(self, dense_csv, dense_df, num_rows):
        num_rows_to_sample = num_rows + 1
        with pytest.raises(
            DrumCommonException,
            match=f"Requested number of rows greater than data length {num_rows_to_sample} > {num_rows}",
        ):
            _ = DrumCLIAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=TargetType.REGRESSION,
                output_dir="path/to/nothing",
                num_rows=num_rows_to_sample,
            ).sample_data_if_necessary(dense_df)


class TestDrumCLIAdapterDenseData(object):
    def test_input_file_contains_X(
        self, dense_csv, dense_df, column_names,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=dense_csv,
            target_type=TargetType.ANOMALY,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, dense_df)
        assert drum_cli_adapter.y is None
        assert drum_cli_adapter.weights is None

        assert np.all(drum_cli_adapter.X.columns == column_names)

    def test_input_file_contains_X_y(
        self, dense_csv_with_target, dense_df, column_names, target_col_name, target_series,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=dense_csv_with_target,
            target_type=TargetType.REGRESSION,
            target_name=target_col_name,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, dense_df)
        pd.testing.assert_series_equal(drum_cli_adapter.y, target_series)
        assert drum_cli_adapter.weights is None

        assert np.all(drum_cli_adapter.X.columns == column_names)
        assert drum_cli_adapter.y.name == target_col_name

    def test_input_file_contains_X_y_weights(
        self,
        dense_csv_with_target_and_weights,
        dense_df,
        column_names,
        target_col_name,
        target_series,
        weights_col_name,
        weights_series,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=dense_csv_with_target_and_weights,
            target_type=TargetType.REGRESSION,
            target_name=target_col_name,
            weights_name=weights_col_name,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, dense_df)
        pd.testing.assert_series_equal(drum_cli_adapter.y, target_series)
        pd.testing.assert_series_equal(drum_cli_adapter.weights, weights_series)

        assert np.all(drum_cli_adapter.X.columns == column_names)
        assert drum_cli_adapter.y.name == target_col_name
        assert drum_cli_adapter.weights.name == weights_col_name


class TestDrumCLIAdapterSparseData(object):
    def test_sparse_input_file_contains_X(
        self, sparse_mtx, sparse_df, sparse_column_names_file, column_names,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=sparse_mtx,
            target_type=TargetType.ANOMALY,
            sparse_column_filename=sparse_column_names_file,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, sparse_df)
        assert drum_cli_adapter.y is None
        assert drum_cli_adapter.weights is None

        assert np.all(drum_cli_adapter.X.columns == column_names)

    def test_sparse_input_file_contains_X_y(
        self,
        sparse_mtx,
        sparse_df,
        sparse_column_names_file,
        column_names,
        target_csv,
        target_series,
        target_col_name,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=sparse_mtx,
            target_type=TargetType.REGRESSION,
            target_filename=target_csv,
            sparse_column_filename=sparse_column_names_file,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, sparse_df)
        pd.testing.assert_series_equal(drum_cli_adapter.y, target_series)
        assert drum_cli_adapter.weights is None

        assert np.all(drum_cli_adapter.X.columns == column_names)
        assert drum_cli_adapter.y.name == target_col_name

    def test_sparse_input_file_contains_X_y_weights(
        self,
        sparse_mtx,
        sparse_df,
        sparse_column_names_file,
        column_names,
        target_csv,
        target_series,
        target_col_name,
        weights_csv,
        weights_series,
        weights_col_name,
    ):
        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=sparse_mtx,
            target_type=TargetType.REGRESSION,
            target_filename=target_csv,
            weights_filename=weights_csv,
            sparse_column_filename=sparse_column_names_file,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, sparse_df)
        pd.testing.assert_series_equal(drum_cli_adapter.y, target_series)
        pd.testing.assert_series_equal(drum_cli_adapter.weights, weights_series)

        assert np.all(drum_cli_adapter.X.columns == column_names)
        assert drum_cli_adapter.y.name == target_col_name
        assert drum_cli_adapter.weights.name == weights_col_name


class TestDrumCLIAdapterSampling(object):
    @pytest.mark.parametrize("num_rows_to_sample", [1, 4.0, 10, "ALL", "num_rows"])
    @pytest.mark.parametrize(
        "input_filename, input_df", [("dense_csv", "dense_df"), ("sparse_mtx", "sparse_df"),]
    )
    def test_sampling(
        self,
        request,
        input_filename,
        input_df,
        target_series,
        weights_series,
        num_rows,
        num_cols,
        col_data,
        target_data_offset,
        weights_data_offset,
        column_names,
        target_col_name,
        weights_col_name,
        num_rows_to_sample,
    ):
        """
        Test for sampling reproducibility
        """
        if num_rows_to_sample == "num_rows":
            num_rows_to_sample = num_rows

        expected_num_rows_sampled = num_rows_to_sample
        if num_rows_to_sample == "ALL":
            expected_num_rows_sampled = num_rows

        input_filename = request.getfixturevalue(input_filename)
        input_df = request.getfixturevalue(input_df)

        drum_cli_adapter = DrumCLIAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=input_filename,
            target_type=TargetType.ANOMALY,
            num_rows=num_rows_to_sample,
        )

        sampled_df = drum_cli_adapter.sample_data_if_necessary(input_df)
        sampled_target = drum_cli_adapter.sample_data_if_necessary(target_series)
        sampled_weights = drum_cli_adapter.sample_data_if_necessary(weights_series)

        # Ensure sparsity never changed
        assert is_sparse_dataframe(sampled_df) == is_sparse_dataframe(input_df)

        # Ensure lengths are equal to num_rows
        assert len(sampled_df) == expected_num_rows_sampled
        assert len(sampled_target) == expected_num_rows_sampled
        assert len(sampled_weights) == expected_num_rows_sampled

        # Ensure column names stayed intact
        assert np.all(sampled_df.columns == column_names)
        assert sampled_target.name == target_col_name
        assert sampled_weights.name == weights_col_name

        # Ensure indices are the same for each sampled df/series
        assert np.all(sampled_df.index == sampled_target.index)
        assert np.all(sampled_df.index == sampled_weights.index)

        # Ensure sampling was performed with replace set to False - no indices should repeat
        assert len(set(sampled_df.index.values)) == len(sampled_df)

        # Ensure values are the same for each sampled df/series with respect to row index
        expected_sampled_col_values = sampled_df.iloc[:, 0]
        if is_sparse_dataframe(sampled_df):
            expected_sampled_col_values = sampled_df.sparse.to_dense().iloc[:, 0]

        expected_sampled_target_values = expected_sampled_col_values + target_data_offset
        expected_sampled_weights_values = expected_sampled_col_values + weights_data_offset

        for i in range(num_cols):
            # Compared all cols to col0 (works for both sparse and dense check)
            pd.testing.assert_series_equal(
                sampled_df.iloc[:, i], sampled_df.iloc[:, 0], check_names=False
            )

        pd.testing.assert_series_equal(
            sampled_target, expected_sampled_target_values, check_names=False
        )
        pd.testing.assert_series_equal(
            sampled_weights, expected_sampled_weights_values, check_names=False
        )

        # Specific test-case, when num_rows is set to ALL, data should not be sampled (and not shuffled)
        if num_rows_to_sample == "ALL":
            pd.testing.assert_series_equal(
                expected_sampled_col_values, pd.Series(col_data), check_names=False
            )
