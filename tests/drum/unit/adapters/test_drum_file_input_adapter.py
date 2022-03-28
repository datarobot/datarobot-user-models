import numpy as np
import pandas as pd
import pytest
from datarobot_drum.drum.adapters.cli.shared.drum_input_file_adapter import DrumInputFileAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe
from datarobot_drum.drum.utils.dataframe import is_sparse_series


class TestDrumInputFileAdapterDenseData(object):
    @pytest.mark.parametrize(
        "input_filename, expected_df",
        [
            ("dense_csv", "dense_df"),
            ("dense_csv_with_target", "dense_df_with_target"),
            ("dense_csv_with_target_and_weights", "dense_df_with_target_and_weights"),
        ],
    )
    def test_dense_input_file_is_read_correctly(
        self, request, input_filename, expected_df,
    ):
        input_filename = request.getfixturevalue(input_filename)
        expected_df = request.getfixturevalue(expected_df)

        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=input_filename, target_type=TargetType.ANOMALY,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.input_dataframe, expected_df)

    def test_input_file_contains_X(
        self, dense_csv, dense_df, column_names,
    ):
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=dense_csv, target_type=TargetType.ANOMALY,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.X, dense_df)
        assert drum_cli_adapter.y is None
        assert drum_cli_adapter.weights is None

        assert np.all(drum_cli_adapter.X.columns == column_names)

    def test_input_file_contains_X_y(
        self, dense_csv_with_target, dense_df, column_names, target_col_name, target_series,
    ):
        drum_cli_adapter = DrumInputFileAdapter(
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
        drum_cli_adapter = DrumInputFileAdapter(
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


class TestDrumInputFileAdapterSparseData(object):
    def test_sparse_column_names_are_read_correctly(self, column_names, sparse_column_names_file):
        sparse_column_names = DrumInputFileAdapter(
            target_type=TargetType.ANOMALY, sparse_column_filename=sparse_column_names_file,
        ).sparse_column_names

        assert sparse_column_names == column_names

    def test_sparse_input_file_is_read_correctly(
        self, sparse_mtx, sparse_df, sparse_column_names_file
    ):
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=sparse_mtx,
            sparse_column_filename=sparse_column_names_file,
            target_type=TargetType.ANOMALY,
        )

        pd.testing.assert_frame_equal(drum_cli_adapter.input_dataframe, sparse_df)

    def test_sparse_input_file_contains_X(
        self, sparse_mtx, sparse_df, sparse_column_names_file, column_names,
    ):
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=sparse_mtx,
            target_type=TargetType.ANOMALY,
            sparse_column_filename=sparse_column_names_file,
        )

        assert is_sparse_dataframe(drum_cli_adapter.X)

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
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=sparse_mtx,
            target_type=TargetType.REGRESSION,
            target_filename=target_csv,
            sparse_column_filename=sparse_column_names_file,
        )

        assert is_sparse_dataframe(drum_cli_adapter.X)
        assert not is_sparse_series(drum_cli_adapter.y)

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
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=sparse_mtx,
            target_type=TargetType.REGRESSION,
            target_filename=target_csv,
            weights_filename=weights_csv,
            sparse_column_filename=sparse_column_names_file,
        )

        assert is_sparse_dataframe(drum_cli_adapter.X)
        assert not is_sparse_series(drum_cli_adapter.y)
        assert not is_sparse_series(drum_cli_adapter.weights)

        pd.testing.assert_frame_equal(drum_cli_adapter.X, sparse_df)
        pd.testing.assert_series_equal(drum_cli_adapter.y, target_series)
        pd.testing.assert_series_equal(drum_cli_adapter.weights, weights_series)

        assert np.all(drum_cli_adapter.X.columns == column_names)
        assert drum_cli_adapter.y.name == target_col_name
        assert drum_cli_adapter.weights.name == weights_col_name


class TestDrumInputFileAdapterBinaryDataProperties(object):
    def test_input_filename_setter_and_lazy_loaded_binary_data(self, dense_csv):
        drum_cli_adapter = DrumInputFileAdapter(
            input_filename=dense_csv, target_type=TargetType.ANOMALY,
        )

        # Lazy load the input binary_data and mimetype by calling input_binary_data, ensure lazy loaded works
        _ = drum_cli_adapter.input_binary_data
        assert drum_cli_adapter._input_binary_data is not None
        assert (
            drum_cli_adapter._input_binary_mimetype is None
        )  # dense mimetype is None, TODO: update!
        assert id(drum_cli_adapter._input_binary_data) == id(drum_cli_adapter.input_binary_data)
        assert id(drum_cli_adapter._input_binary_mimetype) == id(
            drum_cli_adapter.input_binary_mimetype
        )
