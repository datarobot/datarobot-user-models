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
            else:  # CSV format
                df = pd.read_csv(io.BytesIO(binary_data))

                # If the DataFrame only contains a single column, treat blank lines as NANs
                if df.shape[1] == 1:
                    logger.info(
                        "Input data only contains a single column, treating blank lines as NaNs"
                    )
                    df = pd.read_csv(io.BytesIO(binary_data), skip_blank_lines=False)

                return df

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
