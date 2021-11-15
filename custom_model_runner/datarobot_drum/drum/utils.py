"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import io
import json
import logging
import os
import socket
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import BaseLoader, DebugUndefined, Environment
from scipy.io import mmread

from datarobot_drum.drum.common import get_pyarrow_module
from datarobot_drum.drum.enum import (
    ArgumentOptionsEnvVars,
    InputFormatExtension,
    InputFormatToMimetype,
    LOGGER_NAME_PREFIX,
    PredictionServerMimetypes,
)
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class DrumUtils:
    current_path = os.path.dirname(__file__)
    resource_path = os.path.abspath(os.path.join(current_path, "..", "resource"))
    assert os.path.exists(resource_path)

    def __init__(self):
        pass

    @classmethod
    def get_components_repo(cls):
        components_repo = os.path.join(cls.resource_path, "components")
        logger.debug("Components repo: {}".format(components_repo))
        return components_repo

    @classmethod
    def get_pipeline_filepath(cls, pipeline_name):
        pipeline_filepath = os.path.join(cls.resource_path, "pipelines", pipeline_name)
        logger.debug("Getting pipeline: {}".format(pipeline_filepath))
        return pipeline_filepath

    @classmethod
    def render_template_keep_undefined(cls, template_str, data):
        env = Environment(loader=BaseLoader, undefined=DebugUndefined)
        env.filters["jsonify"] = partial(json.dumps, default=str)
        rtemplate = env.from_string(template_str)
        return rtemplate.render(data)

    @classmethod
    def render_file(cls, filename, data):
        with open(filename) as f:
            file_str = f.read()
        return cls.render_template_keep_undefined(file_str, data)

    @classmethod
    def endswith_extension_ignore_case(cls, filename, extensions):
        if isinstance(extensions, list):
            exts = tuple([extension.lower() for extension in extensions])
        elif isinstance(extensions, str):
            exts = extensions.lower()
        else:
            raise ValueError(
                "`extensions` variable type is not supported: {}".format(type(extensions))
            )
        return filename.lower().endswith(exts)

    @classmethod
    def find_files_by_extensions(cls, dirpath, extensions):
        """
        Find files in a dirpath according to provided extensions.
        Ignore case.

        :param dirpath: a directory where to search for files
        :param extensions: list of extensions to match
        :return: list of found files
        """
        lst = []
        for filename in os.listdir(dirpath):
            path = os.path.join(dirpath, filename)
            if os.path.isdir(path):
                continue
            if DrumUtils.endswith_extension_ignore_case(filename, extensions):
                lst.append(path)
        return lst

    @classmethod
    def filename_exists_and_is_file(cls, dirpath, filename, *args):
        filenames = [filename] + list(args)
        abs_filepaths = [os.path.abspath(os.path.join(dirpath, name)) for name in filenames]
        found = [
            os.path.exists(abs_filepath) and os.path.isfile(abs_filepath)
            for abs_filepath in abs_filepaths
        ]

        # Check if there are multiple expected files in the dir, e.g. custom.r and custom.R
        if found.count(True) > 1:
            matching_files = [abs_filepaths[i] for i, value in enumerate(found) if value]
            logger.warning(
                "Found filenames that case-insensitively match each other.\n"
                "Files: {}\n"
                "This could lead to unexpected behavior or errors.".format(matching_files)
            )

        ret = any(found)
        # if no expected files are found, check if there may be files that doesn't match expected case
        # e.g. custom.PY vs custom.PY
        if not ret:
            case_insensitive_names = []
            filenames_lower = [name.lower() for name in filenames]
            for filename in os.listdir(dirpath):
                if filename.lower() in filenames_lower:
                    case_insensitive_names.append(filename)
            if len(case_insensitive_names):
                logger.warning(
                    "Found filenames that case-insensitively match expected filenames.\n"
                    "Found: {}\n"
                    "Expected one of: {}\n"
                    "This could lead to unexpected behavior or errors.".format(
                        case_insensitive_names, filenames
                    )
                )
        return ret

    @classmethod
    def is_port_in_use(cls, host, port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            return s.connect_ex((host, port)) == 0

    @classmethod
    def find_free_port(cls):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @classmethod
    def replace_cmd_argument_value(cls, cmd_list, arg_name, arg_value):
        try:
            ind = cmd_list.index(arg_name)
        except ValueError:
            return
        cmd_list[ind + 1] = arg_value

    @classmethod
    def delete_cmd_argument(cls, cmd_list, arg_name):
        try:
            ind = cmd_list.index(arg_name)
            # Handle case when no value argument, like --skip-deps-install,
            # has to be deleted and it is the last in the list
            if len(cmd_list) == ind + 1 or cmd_list[ind + 1].startswith("--"):
                del cmd_list[ind : ind + 1]
            else:
                del cmd_list[ind : ind + 2]
        except ValueError:
            return


def make_sure_artifact_is_small(output_dir):
    MEGABYTE = 1024 * 1024
    GIGABYTE = 1024 * MEGABYTE
    root_directory = Path(output_dir)
    dir_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())
    logger.info("Artifact directory has been filled to {} Megabytes".format(dir_size / MEGABYTE))
    assert dir_size < 10 * GIGABYTE


def shared_fit_preprocessing(fit_class):
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
    # read in data
    if DrumUtils.endswith_extension_ignore_case(fit_class.input_filename, InputFormatExtension.MTX):
        colnames = None
        if fit_class.sparse_column_file:
            colnames = [column.strip() for column in open(fit_class.sparse_column_file).readlines()]
        df = pd.DataFrame.sparse.from_spmatrix(mmread(fit_class.input_filename), columns=colnames)
    else:
        df = pd.read_csv(fit_class.input_filename)

    # get num rows to use
    if fit_class.num_rows == "ALL":
        fit_class.num_rows = len(df)
    else:
        if fit_class.num_rows > len(df):
            raise DrumCommonException(
                "Requested number of rows greater than data length {} > {}".format(
                    fit_class.num_rows, len(df)
                )
            )
        fit_class.num_rows = int(fit_class.num_rows)

    # get target and features, resample and modify nrows if needed
    if fit_class.target_filename or fit_class.target_name:
        if fit_class.target_filename:
            y_unsampled = pd.read_csv(fit_class.target_filename, index_col=False)
            assert (
                len(y_unsampled.columns) == 1
            ), "Your target dataset at path {} has {} columns named {}".format(
                fit_class.target_filename, len(y_unsampled.columns), y_unsampled.columns
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
            fit_class.target_name = y_unsampled.columns.values[0]
        df = df.dropna(subset=[fit_class.target_name])
        X = df.drop(fit_class.target_name, axis=1).sample(fit_class.num_rows, random_state=1)
        y = df[fit_class.target_name].sample(fit_class.num_rows, random_state=1)

    else:
        X = df.sample(fit_class.num_rows, random_state=1)
        y = None

    parameters = None
    if fit_class.parameter_file:
        parameters = json.load(open(fit_class.parameter_file))

    row_weights = extract_weights(X, fit_class)
    class_order = extract_class_order(fit_class)
    return X, y, class_order, row_weights, parameters


def extract_weights(X, fit_class):
    # extract weights from file or data
    if fit_class.weights_filename:
        row_weights = pd.read_csv(fit_class.weights_filename).sample(
            fit_class.num_rows, random_state=1
        )
    elif fit_class.weights:
        if fit_class.weights not in X.columns:
            raise ValueError(
                "The column name {} is not one of the columns in "
                "your training data".format(fit_class.weights)
            )
        row_weights = X[fit_class.weights]
    else:
        row_weights = None
    return row_weights


def extract_class_order(fit_class):
    # get class order obj from class labels
    if fit_class.negative_class_label is not None and fit_class.positive_class_label is not None:
        class_order = [fit_class.negative_class_label, fit_class.positive_class_label]
    elif fit_class.class_labels:
        class_order = fit_class.class_labels
    else:
        class_order = None

    return class_order


def handle_missing_colnames(df):
    missing_cols = [c for c in df.columns if "Unnamed" in c]
    if missing_cols:
        r_vals = ["X"] + ["X.{}".format(x) for x in range(1, len(missing_cols))]
        missing_lookup = {pycol: rcol for pycol, rcol in zip(missing_cols, r_vals)}
        return df.rename(columns=missing_lookup)
    return df


def unset_drum_supported_env_vars(additional_unset_vars=[]):
    for env_var_key in (
        ArgumentOptionsEnvVars.VALUE_VARS + ArgumentOptionsEnvVars.BOOL_VARS + additional_unset_vars
    ):
        os.environ.pop(env_var_key, None)


class StructuredInputReadUtils:
    @staticmethod
    def read_structured_input_file_as_binary(filename):
        mimetype = StructuredInputReadUtils.resolve_mimetype_by_filename(filename)
        with open(filename, "rb") as file:
            binary_data = file.read()
        return binary_data, mimetype

    @staticmethod
    def read_structured_input_file_as_df(filename, sparse_column_file=None):
        binary_data, mimetype = StructuredInputReadUtils.read_structured_input_file_as_binary(
            filename
        )
        if sparse_column_file:
            with open(sparse_column_file, "rb") as file:
                sparse_colnames = file.read()
        else:
            sparse_colnames = None
        return StructuredInputReadUtils.read_structured_input_data_as_df(
            binary_data, mimetype, sparse_colnames
        )

    @staticmethod
    def resolve_mimetype_by_filename(filename):
        return InputFormatToMimetype.get(os.path.splitext(filename)[1])

    @staticmethod
    def read_structured_input_data_as_df(binary_data, mimetype, sparse_colnames=None):
        try:
            if mimetype == PredictionServerMimetypes.TEXT_MTX:
                columns = None
                if sparse_colnames:
                    columns = [
                        column.strip().decode("utf-8")
                        for column in io.BytesIO(sparse_colnames).readlines()
                    ]
                return pd.DataFrame.sparse.from_spmatrix(
                    mmread(io.BytesIO(binary_data)), columns=columns
                )
            elif mimetype == PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM:
                df = get_pyarrow_module().ipc.deserialize_pandas(binary_data)

                # After CSV serialization+deserialization,
                # original dataframe's None and np.nan values
                # become np.nan values.
                # After Arrow serialization+deserialization,
                # original dataframe's None and np.nan values
                # become np.nan for numeric columns and None for 'object' columns.
                #
                # Since we are supporting both CSV and Arrow,
                # to be consistent with CSV serialization/deserialization,
                # it is required to replace all None with np.nan for Arrow.
                df.fillna(value=np.nan, inplace=True)

                return df
            else:
                return pd.read_csv(io.BytesIO(binary_data))
        except pd.errors.ParserError as e:
            raise DrumCommonException(
                "Pandas failed to read input binary data {}".format(binary_data)
            )


@contextmanager
def capture_R_traceback_if_errors(r_handler, logger):
    from rpy2.rinterface_lib.embedded import RRuntimeError

    try:
        yield
    except RRuntimeError as e:
        try:
            out = "\n".join(r_handler("capture.output(traceback(max.lines = 50))"))
            logger.error("R Traceback:\n{}".format(str(out)))
        except Exception as traceback_exc:
            e.context = {
                "r_traceback": "(an error occurred while getting traceback from R)",
                "t_traceback_err": traceback_exc,
            }
        raise
