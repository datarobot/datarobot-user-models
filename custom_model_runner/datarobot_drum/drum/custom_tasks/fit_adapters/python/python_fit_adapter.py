"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import logging
import sys

import pandas as pd

from datarobot_drum.drum.custom_tasks.fit_adapters.BaseFitAdapter import BaseFitAdapter
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.utils import make_sure_artifact_is_small, StructuredInputReadUtils

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonFitAdapter(BaseFitAdapter):
    def configure(self):
        sys.path.append(self.custom_task_folder_path)

    def _extract_weights(self, X):
        # extract weights from file or data
        if self.weights_filename:
            row_weights = pd.read_csv(self.weights_filename).sample(self.num_rows, random_state=1)
        elif self.weights:
            if self.weights not in X.columns:
                raise ValueError(
                    "The column name {} is not one of the columns in "
                    "your training data".format(self.weights)
                )
            row_weights = X[self.weights]
        else:
            row_weights = None
        return row_weights

    def _extract_class_order(self):
        # get class order obj from class labels
        if self.negative_class_label is not None and self.positive_class_label is not None:
            class_order = [self.negative_class_label, self.positive_class_label]
        elif self.class_labels:
            class_order = self.class_labels
        else:
            class_order = None

        return class_order

    def _python_fit_preprocessing(self):
        """
        Shared preprocessing to get X, y, class_order, row_weights, and parameters.
        Used by _materialize method for both python and R fitting.

        :param fit_class: PythonFit or RFit class
        :return:
            X: pd.DataFrame of features to use in fit
            y: pd.Series of target to use in fit
            class_order: array specifying class order, or None
            row_weights: pd.Series of row weights, or None
        """
        df = StructuredInputReadUtils.read_structured_input_file_as_df(
            filename=self.input_filename, sparse_column_file=self.sparse_column_filename
        )

        # get num rows to use
        if self.num_rows == "ALL":
            self.num_rows = len(df)
        else:
            if self.num_rows > len(df):
                raise DrumCommonException(
                    "Requested number of rows greater than data length {} > {}".format(
                        self.num_rows, len(df)
                    )
                )
            self.num_rows = int(self.num_rows)

        # get target and features, resample and modify nrows if needed
        if self.target_filename or self.target_name:
            if self.target_filename:
                y_unsampled = pd.read_csv(self.target_filename, index_col=False)
                assert (
                    len(y_unsampled.columns) == 1
                ), "Your target dataset at path {} has {} columns named {}".format(
                    self.target_filename, len(y_unsampled.columns), y_unsampled.columns
                )
                assert len(df) == len(
                    y_unsampled
                ), "Your input data has {} entries, but your target data has {} entries".format(
                    len(df), len(y_unsampled)
                )
                if y_unsampled.columns[0] in df.columns:
                    y_unsampled.columns = ["__target__"]
                df = df.merge(y_unsampled, left_index=True, right_index=True)
                assert len(y_unsampled.columns.values) == 1
                self.target_name = y_unsampled.columns.values[0]
            df = df.dropna(subset=[self.target_name])
            X = df.drop(self.target_name, axis=1).sample(self.num_rows, random_state=1)
            y = df[self.target_name].sample(self.num_rows, random_state=1)

        else:
            X = df.sample(self.num_rows, random_state=1)
            y = None

        parameters = None
        if self.parameters_file:
            parameters = json.load(open(self.parameters_file))
        elif self.default_parameter_values:
            parameters = self.default_parameter_values

        row_weights = self._extract_weights(X)
        class_order = self._extract_class_order()
        return X, y, class_order, row_weights, parameters

    def outer_fit(self):
        model_adapter = PythonModelAdapter(self.custom_task_folder_path, self.target_type)
        model_adapter.load_custom_hooks()

        X, y, class_order, row_weights, parameters = self._python_fit_preprocessing()
        model_adapter.fit(
            X,
            y,
            output_dir=self.output_dir,
            class_order=class_order,
            row_weights=row_weights,
            parameters=parameters,
        )

        make_sure_artifact_is_small(self.output_dir)
