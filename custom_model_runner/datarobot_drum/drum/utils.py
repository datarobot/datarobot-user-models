import json
import logging
import os
import socket
from scipy.io import mmread
from contextlib import closing
from functools import partial
from pathlib import Path
from datarobot_drum.drum.exceptions import DrumCommonException

import pandas as pd
from jinja2 import BaseLoader, DebugUndefined, Environment

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class CMRunnerUtils:
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
    def find_files_by_extensions(cls, dirpath, extensions):
        lst = []
        for filename in os.listdir(dirpath):
            path = os.path.join(dirpath, filename)
            if os.path.isdir(path):
                continue
            if any(filename.endswith(extension) for extension in extensions):
                lst.append(path)
        return lst

    @classmethod
    def filename_exists_and_is_file(cls, dirpath, filename):
        abs_filepath = os.path.abspath(os.path.join(dirpath, filename))
        return os.path.exists(abs_filepath) and os.path.isfile(abs_filepath)

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
    Shared preprocessing to get X, y, class_order, and row_weights.
    Used by _materialize method for both python and R fitting.

    :param fit_class: PythonFit or RFit class
    :return:
        X: pd.DataFrame of features to use in fit
        y: pd.Series of target to use in fit
        class_order: array specifying class order, or None
        row_weights: pd.Series of row weights, or None
    """
    # read in data
    if fit_class.input_filename.endswith(".mtx"):
        df = pd.DataFrame.sparse.from_spmatrix(mmread(fit_class.input_filename))
    else:
        df = pd.read_csv(fit_class.input_filename, lineterminator="\n")

    # get num rows to use
    if fit_class.num_rows == "ALL":
        fit_class.num_rows = len(df)
    else:
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
            df = df.merge(y_unsampled, left_index=True, right_index=True)
            assert len(y_unsampled.columns.values) == 1
            fit_class.target_name = y_unsampled.columns.values[0]
        df = df.dropna(subset=[fit_class.target_name])
        X = df.drop(fit_class.target_name, axis=1).sample(
            fit_class.num_rows, random_state=1, replace=True
        )
        y = df[fit_class.target_name].sample(fit_class.num_rows, random_state=1, replace=True)

    else:
        X = df.sample(fit_class.num_rows, random_state=1, replace=True)
        y = None

    row_weights = extract_weights(X, fit_class)
    class_order = extract_class_order(fit_class)
    return X, y, class_order, row_weights


def extract_weights(X, fit_class):
    # extract weights from file or data
    if fit_class.weights_filename:
        row_weights = pd.read_csv(fit_class.weights_filename, lineterminator="\n").sample(
            fit_class.num_rows, random_state=1, replace=True
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
    if fit_class.negative_class_label and fit_class.positive_class_label:
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
