"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys
from typing import Optional

import pandas as pd
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.utils.dataframe import is_sparse_series
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class DrumInputFileAdapter(object):
    """
    Shared parent class for other adapters that handles parsing input file arguments
    """

    def __init__(
        self,
        target_type,
        input_filename=None,
        target_name=None,
        target_filename=None,
        weights_name=None,
        weights_filename=None,
        sparse_column_filename=None,
    ):
        """
        Parameters
        ----------
        target_type: datarobot_drum.drum.enum.TargetType
        input_filename: str or None
            Path to the input training dataset
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
        """
        self.target_type = target_type
        self.input_filename = input_filename
        self.target_name = target_name
        self.target_filename = target_filename
        self.weights_name = weights_name
        self.weights_filename = weights_filename
        self.sparse_column_filename = sparse_column_filename

        # Lazy loaded variables
        self._input_dataframe = None
        self._input_binary_data = None
        self._input_binary_mimetype = None

    def _lazy_load_binary_data(self):
        if self._input_binary_data is None:
            (
                self._input_binary_data,
                self._input_binary_mimetype,
            ) = StructuredInputReadUtils.read_structured_input_file_as_binary(
                filename=self.input_filename
            )
        return self

    @property
    def input_binary_data(self):
        return self._lazy_load_binary_data()._input_binary_data

    @property
    def input_binary_mimetype(self):
        return self._lazy_load_binary_data()._input_binary_mimetype

    @property
    def sparse_column_names(self):
        if self.sparse_column_filename:
            return StructuredInputReadUtils.read_sparse_column_file_as_list(
                self.sparse_column_filename
            )

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
                self.input_filename,
                self.sparse_column_filename,
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
            raise DrumCommonException("Must provide target name or target filename for y")

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
