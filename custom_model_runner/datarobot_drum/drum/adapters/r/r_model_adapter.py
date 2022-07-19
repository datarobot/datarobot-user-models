"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os

import pandas as pd

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe
from datarobot_drum.drum.utils.drum_utils import make_sure_artifact_is_small
from datarobot_drum.drum.utils.stacktraces import capture_R_traceback_if_errors

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_FIT_PATH = os.path.join(CUR_DIR, "fit.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR, "..", "..", "..", "drum", "language_predictors", "r_common_code", "common.R",
    )
)


class RModelAdapter(object):
    R = None
    R_RUNTIME = None
    R_PANDAS = None

    def __init__(
        self, custom_task_folder_path, target_type,
    ):
        """
        Parameters
        ----------
        custom_task_folder_path: str
            Path to the custom task folder
        target_type: datarobot_drum.drum.enum.TargetType
        """
        self.custom_task_folder_path = custom_task_folder_path
        self.target_type = target_type

    def load_custom_hooks(self):
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr

            pandas2ri.activate()
            RModelAdapter.R_PANDAS = pandas2ri

            RModelAdapter.R = ro
            RModelAdapter.R_RUNTIME = ro.r

            RModelAdapter.R_RUNTIME.source(R_COMMON_PATH)
            RModelAdapter.R_RUNTIME.source(R_FIT_PATH)
            RModelAdapter.R_RUNTIME.outer_init(self.custom_task_folder_path, self.target_type.value)
        except ImportError:
            error_message = (
                "rpy2 package is not installed."
                "Install datarobot-drum using 'pip install datarobot-drum[R]'"
                "Available for Python>=3.6"
            )
            logger.error(error_message)
            exit(1)

    def _convert_bool_to_str(self, pd_type):
        """
        R does not like native python bool types, so convert them into strings.
        """
        d = {True: "True", False: "False"}
        if isinstance(pd_type, pd.DataFrame):
            mask = pd_type.applymap(type) != bool
            return pd_type.where(mask, pd_type.replace(d))
        # else, Series
        if pd_type.dtype == bool:
            return pd_type.replace(d)
        return pd_type

    def _convert(self, py_type):
        """
        Converts native python type into R types, to be passed into R functions.
        """
        if isinstance(py_type, pd.DataFrame):
            if is_sparse_dataframe(py_type):
                coo_matrix = py_type.sparse.to_coo()
                r_sparse_matrix = self.R_RUNTIME("Matrix::sparseMatrix")(
                    i=self.R.IntVector(coo_matrix.row + 1),
                    j=self.R.IntVector(coo_matrix.col + 1),
                    x=self.R.FloatVector(coo_matrix.data),
                    dims=self.R.IntVector(list(coo_matrix.shape)),
                    dimnames=self.R.ListVector([("0", self.R.NULL), ("1", list(py_type.columns))]),
                    giveCsparse=0,  # returns triplet format (same as coo)
                )
                return r_sparse_matrix
            else:
                return self.R_PANDAS.py2rpy_pandasdataframe(self._convert_bool_to_str(py_type))
        elif isinstance(py_type, pd.Series):
            return self.R_PANDAS.py2rpy_pandasseries(self._convert_bool_to_str(py_type))
        elif isinstance(py_type, list):
            return self.R.StrVector(py_type)
        elif isinstance(py_type, dict):
            return self.R.ListVector(py_type)
        elif isinstance(py_type, str):
            return py_type
        elif py_type is None:
            return self.R.NULL
        else:
            raise ValueError(f"Error converting python variable of type '{type(py_type)}' to R")

    def fit(self, X, y, output_dir, class_order=None, row_weights=None, parameters=None):
        """
        Trains an R-based custom task.

        Parameters
        ----------
        X: pd.DataFrame
            Training data. Could be sparse or dense
        y: pd.Series
            Target values
        output_dir: str
            Output directory to store the artifact
        class_order: list or None
            Expected order of classification labels
        row_weights: pd.Series or None
            Class weights
        parameters: dict or None
            Hyperparameter values

        Returns
        -------
        RModelAdapter
        """
        # make sure our output dir ends with a slash
        if output_dir[-1] != "/":
            output_dir += "/"

        with capture_R_traceback_if_errors(self.R_RUNTIME, logger):
            self.R_RUNTIME.outer_fit(
                X=self._convert(X),
                y=self._convert(y),
                output_dir=self._convert(output_dir),
                class_order=self._convert(class_order),
                row_weights=self._convert(row_weights),
                parameters=self._convert(parameters),
            )
        make_sure_artifact_is_small(output_dir)
        return self
