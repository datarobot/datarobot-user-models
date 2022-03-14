"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os

from datarobot_drum.drum.custom_tasks.fit_adapters.base_fit_adapter import BaseFitAdapter
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.utils import (
    make_sure_artifact_is_small,
    capture_R_traceback_if_errors,
)

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_FIT_PATH = os.path.join(CUR_DIR, "fit.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR, "..", "..", "..", "..", "drum", "language_predictors", "r_common_code", "common.R",
    )
)


class RFitAdapter(BaseFitAdapter):
    R = None
    R_RUNTIME = None

    def configure(self):
        super(RFitAdapter, self).configure()

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            pandas2ri.activate()

            RFitAdapter.R = ro
            RFitAdapter.R_RUNTIME = ro.r

            RFitAdapter.R_RUNTIME.source(R_COMMON_PATH)
            RFitAdapter.R_RUNTIME.source(R_FIT_PATH)
            RFitAdapter.R_RUNTIME.init(self.custom_task_folder_path)
        except ImportError:
            error_message = (
                "rpy2 package is not installed."
                "Install datarobot-drum using 'pip install datarobot-drum[R]'"
                "Available for Python>=3.6"
            )
            logger.error(error_message)
            exit(1)

    def outer_fit(self):
        r = RFitAdapter.R
        r_runtime = RFitAdapter.R_RUNTIME

        # make sure our output dir ends with a slash
        if self.output_dir[-1] != "/":
            self.output_dir += "/"

        weights = self.weights if self.weights else r.NULL
        target_name = self.target_name if self.target_name else r.NULL

        # TODO: read files in python, pass values to R instead of filepaths
        with capture_R_traceback_if_errors(r_runtime, logger):
            r_runtime.outer_fit(
                self.output_dir,
                self.input_filename,
                self.sparse_column_filename or r.NULL,
                self.target_filename or r.NULL,
                target_name,
                self.num_rows,
                self.weights_filename or r.NULL,
                weights,
                r.NULL if self.positive_class_label is None else self.positive_class_label,
                r.NULL if self.negative_class_label is None else self.negative_class_label,
                r.StrVector([str(l) for l in self.class_labels]) if self.class_labels else r.NULL,
                self.parameters_file or r.NULL,
                self.target_type.value,  # target_type is an enum, pass its string value into R
                r.DataFrame(self.default_parameter_values)
                if self.default_parameter_values
                else r.NULL,
            )
        make_sure_artifact_is_small(self.output_dir)
        return []
