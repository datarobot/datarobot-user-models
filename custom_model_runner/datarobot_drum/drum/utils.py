import os
import logging
import socket
import pandas as pd
from contextlib import closing
from jinja2 import Environment, BaseLoader, DebugUndefined
from pathlib import Path
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
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(
            template_str
        )
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
    df = pd.read_csv(fit_class.input_filename)

    # get num rows to use
    if fit_class.num_rows == "ALL":
        fit_class.num_rows = len(df)
    else:
        fit_class.num_rows = int(fit_class.num_rows)

    # get target and features, resample and modify nrows if needed
    if fit_class.target_filename:
        X = df.sample(fit_class.num_rows, random_state=1)
        y = pd.read_csv(fit_class.target_filename, index_col=False).sample(
            fit_class.num_rows, random_state=1
        )
        assert len(y.columns) == 1
        assert len(X) == len(y)
        y = y.iloc[:, 0]
    else:
        X = df.drop(fit_class.target_name, axis=1).sample(
            fit_class.num_rows, random_state=1, replace=True
        )
        y = df[fit_class.target_name].sample(fit_class.num_rows, random_state=1, replace=True)

    # extract weights from file or data
    if fit_class.weights_filename:
        row_weights = pd.read_csv(fit_class.weights_filename).sample(
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

    # get class order obj from class labels
    class_order = (
        [fit_class.negative_class_label, fit_class.positive_class_label]
        if fit_class.negative_class_label
        else None
    )

    return X, y, class_order, row_weights
