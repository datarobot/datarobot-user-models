"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os

from datarobot_drum.drum.adapters.drum_cli_adapter import DrumCLIAdapter
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.utils.drum_utils import make_sure_artifact_is_small
from datarobot_drum.drum.utils.stacktraces import capture_R_traceback_if_errors

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_FIT_PATH = os.path.join(CUR_DIR, "fit.R")
R_COMMON_PATH = os.path.abspath(
    os.path.join(
        CUR_DIR, "..", "..", "..", "..", "drum", "language_predictors", "r_common_code", "common.R",
    )
)


class RModelAdapter(object):
    R = None
    R_RUNTIME = None

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
            from rpy2.robjects.conversion import localconverter

            pandas2ri.activate()

            RModelAdapter.R = ro
            RModelAdapter.R_RUNTIME = ro.r

            RModelAdapter.R_RUNTIME.source(R_COMMON_PATH)
            RModelAdapter.R_RUNTIME.source(R_FIT_PATH)
            RModelAdapter.R_RUNTIME.init(self.custom_task_folder_path)
        except ImportError:
            error_message = (
                "rpy2 package is not installed."
                "Install datarobot-drum using 'pip install datarobot-drum[R]'"
                "Available for Python>=3.6"
            )
            logger.error(error_message)
            exit(1)

    def fit(self, drum_cli_adapter):
        """
        TODO: decouple drum_cli_adapter from this function. This should match (python) ModelAdapter.fit args
        This will involve transforming python datatypes to R datatypes

        Parameters
        ----------
        drum_cli_adapter: DrumCLIAdapter

        Returns
        -------
        RModelAdapter
        """
        r = RModelAdapter.R
        r_runtime = RModelAdapter.R_RUNTIME

        # make sure our output dir ends with a slash
        if drum_cli_adapter.output_dir[-1] != "/":
            drum_cli_adapter.output_dir += "/"

        # TODO: read files in python, pass values to R instead of filepaths
        with capture_R_traceback_if_errors(r_runtime, logger):
            r_runtime.outer_fit(
                drum_cli_adapter.output_dir,
                drum_cli_adapter.input_filename,
                drum_cli_adapter.sparse_column_filename or r.NULL,
                drum_cli_adapter.target_filename or r.NULL,
                drum_cli_adapter.target_name or r.NULL,
                drum_cli_adapter.num_rows,
                drum_cli_adapter.weights_filename or r.NULL,
                drum_cli_adapter.weights_name or r.NULL,
                r.NULL
                if drum_cli_adapter.positive_class_label is None
                else drum_cli_adapter.positive_class_label,
                r.NULL
                if drum_cli_adapter.negative_class_label is None
                else drum_cli_adapter.negative_class_label,
                r.StrVector([str(l) for l in drum_cli_adapter.class_labels])
                if drum_cli_adapter.class_labels
                else r.NULL,
                drum_cli_adapter.parameters_file or r.NULL,
                drum_cli_adapter.target_type.value,  # target_type is an enum, pass its string value into R
                r.DataFrame(drum_cli_adapter.default_parameter_values)
                if drum_cli_adapter.default_parameter_values
                else r.NULL,
            )
        make_sure_artifact_is_small(drum_cli_adapter.output_dir)
        return self
