"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import logging
import shutil
import sys
from tempfile import mkdtemp
from typing import Optional

import pandas as pd

from datarobot_drum.drum.adapters.classification_labels_util import (
    needs_class_labels,
    infer_class_labels,
)
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, TargetType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.dataframe import is_sparse_series
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class DrumCLIAdapter(object):
    """
    This class acts as a layer between the DRUM CLI and running a custom task's fit method. It will convert the
    cli arguments (mostly paths) into tangible variables to be passed into fit.
    """

    def __init__(
        self,
        custom_task_folder_path,
        input_filename,
        target_type,
        target_name=None,
        target_filename=None,
        weights_name=None,
        weights_filename=None,
        sparse_column_filename=None,
        positive_class_label=None,
        negative_class_label=None,
        class_labels=None,
        parameters_file=None,
        default_parameter_values=None,
        output_dir=None,
        num_rows=None,
    ):
        """
        Parameters
        ----------
        custom_task_folder_path: str
            Path to the custom task folder
        input_filename: str
            Path to the input training dataset
        target_type: datarobot_drum.drum.enum.TargetType
        target_name: str or None
            Optional. Name of the target column in the input training dataset
        target_filename: str or None
            Optional. Path to the target values if it's not included in the input training dataset
        weights_name: str or None
            Optional. Name of the weights column in the input training dataset
        weights_filename: str or None
            Optional. Path to the weight values if it's not included in the input training dataset
        sparse_column_filename: str or None
            Optional. Path to the column names if it's not included in the input training dataset
        positive_class_label: str or None
            Optional. Name of the positive class label if target type is binary
        negative_class_label: str or None
            Optional. Name of the negative class label if target type is binary
        class_labels: list[str] or None
            Optional. List of class labels
        parameters_file: str or None
            Optional. Path to the hyperparameter values
        default_parameter_values: dict[str, Any] or None
            Optional. Dict containing default parameter values from the model metadata
        output_dir: str or None
            Optional. Output directory to store the fit artifacts
        num_rows: int or None
            Optional. Number of rows
        """
        self.custom_task_folder_path = custom_task_folder_path
        self.input_filename = input_filename
        self.target_type = target_type
        self.target_name = target_name
        self.target_filename = target_filename
        self.weights_name = weights_name
        self.weights_filename = weights_filename
        self.sparse_column_filename = sparse_column_filename
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.class_labels = class_labels
        self.parameters_file = parameters_file
        self.default_parameter_values = default_parameter_values
        self.output_dir = output_dir
        self.num_rows = num_rows

        self.persist_output = self.output_dir is not None

        # Lazy loaded variables
        self._input_dataframe = None

    @property
    def input_dataframe(self):
        """
        Returns
        -------
        pandas.DataFrame
            Raw input data file as a dataframe. Could be dense/sparse, include/not include the target, or include/not
            include weights.
        """
        if self._input_dataframe is None:
            # Lazy load df
            self._input_dataframe = StructuredInputReadUtils.read_structured_input_file_as_df(
                self.input_filename, self.sparse_column_filename,
            )
        return self._input_dataframe

    @property
    def X(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            Input training data, with no target included.
        """
        X = self.input_dataframe
        cols_to_drop = []

        if self.target_name:
            cols_to_drop.append(self.target_name)
        if self.weights_name:
            cols_to_drop.append(self.weights_name)

        if cols_to_drop:
            X = X.drop(cols_to_drop, axis=1)

        return X

    @property
    def y(self) -> Optional[pd.Series]:
        """
        Returns
        -------
        pd.Series or None
        """
        if self.target_type == TargetType.ANOMALY:
            return None
        elif self.target_name:
            assert self.target_filename is None

            if self.target_name not in self.input_dataframe.columns:
                e = "The target column '{}' does not exist in your input data.".format(
                    self.target_name
                )
                print(e, file=sys.stderr)
                raise DrumCommonException(e)

            y = self.input_dataframe[self.target_name]
            if is_sparse_series(y):
                return y.sparse.to_dense()
            return y
        elif self.target_filename:
            assert self.target_name is None

            y = pd.read_csv(self.target_filename, index_col=False)
            return y.iloc[:, 0]
        elif self.target_type == TargetType.TRANSFORM:
            return None  # It is valid for a transform to have no target (ie when the project target type is anomaly)
        else:
            raise DrumCommonException("Must provide target name or target filename to drum fit")

    @property
    def weights(self) -> Optional[pd.Series]:
        """
        Returns
        -------
        pd.Series or None
        """
        if self.weights_filename:
            weights = pd.read_csv(self.weights_filename, index_col=False).iloc[:, 0]
        elif self.weights_name:
            if self.weights_name not in self.input_dataframe.columns:
                raise DrumCommonException(
                    "The weights column '{}' does not exist in your input data.".format(
                        self.weights_name
                    )
                )
            weights = self.input_dataframe[self.weights_name]
            if is_sparse_series(weights):
                return weights.sparse.to_dense()
            return weights
        else:
            weights = None
        return weights

    @property
    def class_ordering(self):
        # get class order obj from class labels
        if self.negative_class_label is not None and self.positive_class_label is not None:
            class_order = [self.positive_class_label, self.negative_class_label]
        elif self.class_labels:
            class_order = self.class_labels
        else:
            class_order = None

        return class_order

    @property
    def parameters(self):
        if self.parameters_file:
            return json.load(open(self.parameters_file))
        return {}

    @property
    def default_parameters(self):
        if self.default_parameter_values:
            return self.default_parameter_values
        return {}

    @property
    def parameters_for_fit(self):
        if self.parameters:
            return self.parameters
        return self.default_parameters

    def _infer_class_labels_if_not_provided(self):
        if needs_class_labels(
            target_type=self.target_type,
            negative_class_label=self.negative_class_label,
            positive_class_label=self.positive_class_label,
            class_labels=self.class_labels,
        ):
            # TODO: Only pass y in here
            class_labels = infer_class_labels(
                target_type=self.target_type,
                input_filename=self.input_filename,
                target_filename=self.target_filename,
                target_name=self.target_name,
            )

            if self.target_type == TargetType.BINARY:
                self.positive_class_label, self.negative_class_label = class_labels
            elif self.target_type == TargetType.MULTICLASS:
                self.class_labels = class_labels

    def _validate_output_dir(self):
        """
        Validate that the output directory is not the same as the input.
        If no output dir exists, then assign a temporary directory to it.

        Returns
        -------
        DrumCLIAdapter
            self

        Raises
        ------
        DrumCommonException
            Raised when the code directory is also used as the output directory.
        """
        if self.output_dir == self.custom_task_folder_path:
            raise DrumCommonException("The code directory may not be used as the output directory.")

        if not self.persist_output:
            self.output_dir = mkdtemp()

        return self

    def validate(self):
        """
        After initialization, validate and configure the inputs.

        Returns
        -------
        DrumCLIAdapter
        """
        self._validate_output_dir()
        self._infer_class_labels_if_not_provided()

        return self

    def sample_data_if_necessary(self, data):
        if self.num_rows == "ALL":
            return data

        assert isinstance(self.num_rows, (int, float))
        self.num_rows = int(self.num_rows)
        if self.num_rows > len(self.input_dataframe):
            raise DrumCommonException(
                "Requested number of rows greater than data length {} > {}".format(
                    self.num_rows, len(self.input_dataframe)
                )
            )

        return data.sample(self.num_rows, random_state=1)

    def cleanup_output_directory_if_necessary(self):
        """
        Returns
        -------
        bool
            True if output directory was cleaned up. False otherwise.
        """
        if not self.persist_output:
            shutil.rmtree(self.output_dir)
        return not self.persist_output
