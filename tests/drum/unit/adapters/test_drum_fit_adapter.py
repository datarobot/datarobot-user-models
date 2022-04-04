import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from datarobot_drum.drum.adapters.cli.drum_fit_adapter import DrumFitAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe


class TestDrumFitAdapterFailures(object):
    @pytest.mark.parametrize(
        "target_type", [TargetType.BINARY, TargetType.REGRESSION, TargetType.MULTICLASS],
    )
    def test_target_data_missing(self, dense_csv, target_type):
        with pytest.raises(
            DrumCommonException, match="Must provide target name or target filename for y"
        ):
            _ = DrumFitAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=target_type,
            ).y

    @pytest.mark.parametrize(
        "target_type", [TargetType.ANOMALY, TargetType.TRANSFORM],
    )
    def test_target_data_missing_okay(self, dense_csv, target_type):
        y = DrumFitAdapter(
            custom_task_folder_path="path/to/nothing",
            input_filename=dense_csv,
            target_type=target_type,
        ).y

        assert y is None

    def test_input_file_contains_X_missing_y(
        self, dense_csv, target_col_name,
    ):
        with pytest.raises(
            DrumCommonException,
            match=f"The target column '{target_col_name}' does not exist in your input data.",
        ):
            _ = DrumFitAdapter(
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
            _ = DrumFitAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=TargetType.REGRESSION,
                weights_name=weights_col_name,
            ).weights

    def test_output_dir_cannot_equal_custom_task_folder_path(self):
        with pytest.raises(
            DrumCommonException, match="The code directory may not be used as the output directory."
        ):
            _ = DrumFitAdapter(
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
            _ = DrumFitAdapter(
                custom_task_folder_path="path/to/nothing",
                input_filename=dense_csv,
                target_type=TargetType.REGRESSION,
                output_dir="path/to/nothing",
                num_rows=num_rows_to_sample,
            ).sample_data_if_necessary(dense_df)


class TestDrumFitAdapterSampling(object):
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

        drum_cli_adapter = DrumFitAdapter(
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


class TestDrumFitAdapterParameters(object):
    @pytest.fixture
    def parameters(self):
        return {"param_1": 1, "param_2": 2}

    @pytest.fixture
    def parameters_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
            with open(temp_file.name, "w") as f:
                json.dump({"param_1": 1, "param_2": 2}, f)

            yield temp_file.name

    @pytest.fixture
    def default_parameter_values(self):
        return {"param_1": 0, "param_2": 0}

    def test_parameters_default_to_empty_dict(self):
        drum_cli_adapter = DrumFitAdapter(
            custom_task_folder_path="a/path", target_type=TargetType.REGRESSION,
        )

        assert drum_cli_adapter.parameters == {}
        assert drum_cli_adapter.default_parameters == {}
        assert drum_cli_adapter.parameters_for_fit == {}

    def test_parameters_used_for_fit_if_file_provided(
        self, parameters, parameters_file, default_parameter_values
    ):
        drum_cli_adapter = DrumFitAdapter(
            custom_task_folder_path="a/path",
            target_type=TargetType.REGRESSION,
            parameters_file=parameters_file,
            default_parameter_values=default_parameter_values,
        )

        assert drum_cli_adapter.parameters == parameters
        assert drum_cli_adapter.default_parameters == default_parameter_values
        assert drum_cli_adapter.parameters_for_fit == parameters

    def test_default_parameters_used_for_fit_if_file_not_provided(self, default_parameter_values):
        drum_cli_adapter = DrumFitAdapter(
            custom_task_folder_path="a/path",
            target_type=TargetType.REGRESSION,
            default_parameter_values=default_parameter_values,
        )

        assert drum_cli_adapter.parameters == {}
        assert drum_cli_adapter.default_parameters == default_parameter_values
        assert drum_cli_adapter.parameters_for_fit == default_parameter_values


class TestDrumFitAdapterOutputDir(object):
    def test_temp_output_dir_set_when_provided(self):
        output_dir = "output/path"
        drum_cli_adapter = DrumFitAdapter(
            custom_task_folder_path="a/path",
            target_type=TargetType.REGRESSION,
            output_dir=output_dir,
        )._validate_output_dir()

        assert drum_cli_adapter.output_dir == output_dir
        assert drum_cli_adapter.persist_output
        assert not drum_cli_adapter.cleanup_output_directory_if_necessary()

    def test_temp_output_dir_created_and_cleaned_up_when_not_provided(self):
        drum_cli_adapter = DrumFitAdapter(
            custom_task_folder_path="a/path", target_type=TargetType.REGRESSION,
        )._validate_output_dir()

        # Ensure output is not to be persisted
        assert not drum_cli_adapter.persist_output

        # Ensure output_dir is pointing to a temporary directory
        assert os.path.isdir(drum_cli_adapter.output_dir)

        # Ensure temp directory gets cleaned up
        assert drum_cli_adapter.cleanup_output_directory_if_necessary()
        assert not os.path.isdir(drum_cli_adapter.output_dir)
