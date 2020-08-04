import logging
import os
import sys
from contextlib import contextmanager
from enum import Enum

LOGGER_NAME_PREFIX = "drum"
REGRESSION_PRED_COLUMN = "Predictions"
CUSTOM_FILE_NAME = "custom"
POSITIVE_CLASS_LABEL_ARG_KEYWORD = "positive_class_label"
NEGATIVE_CLASS_LABEL_ARG_KEYWORD = "negative_class_label"

URL_PREFIX_ENV_VAR_NAME = "URL_PREFIX"

LOG_LEVELS = {
    "noset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class SupportedFrameworks(object):
    SKLEARN = "scikit-learn"
    TORCH = "torch"
    KERAS = "keras"
    XGBOOST = "xgboost"
    PYPMML = "pypmml"


extra_deps = {
    SupportedFrameworks.SKLEARN: ["scikit-learn", "scipy", "numpy"],
    SupportedFrameworks.TORCH: ["torch", "numpy", "scikit-learn", "scipy"],
    SupportedFrameworks.KERAS: ["scipy", "numpy", "h5py", "keras", "tensorflow"],
    SupportedFrameworks.XGBOOST: ["scipy", "numpy", "xgboost"],
    SupportedFrameworks.PYPMML: ["pypmml"],
}


class CustomHooks(object):
    INIT = "init"
    LOAD_MODEL = "load_model"
    TRANSFORM = "transform"
    SCORE = "score"
    POST_PROCESS = "post_process"
    FIT = "fit"

    ALL_PREDICT = [INIT, LOAD_MODEL, TRANSFORM, SCORE, POST_PROCESS]
    ALL = ALL_PREDICT + [FIT]


class PythonArtifacts(object):
    PKL_EXTENSION = ".pkl"
    TORCH_EXTENSION = ".pth"
    KERAS_EXTENSION = ".h5"
    JOBLIB_EXTENSION = ".joblib"
    PYPMML_EXTENSION = ".pmml"
    ALL = [PKL_EXTENSION, TORCH_EXTENSION, KERAS_EXTENSION, JOBLIB_EXTENSION, PYPMML_EXTENSION]


class RArtifacts(object):
    RDS_EXTENSION = ".rds"
    ALL = [RDS_EXTENSION]


class JavaArtifacts(object):
    JAR_EXTENSION = ".jar"
    MOJO_EXTENSION = ".zip"
    POJO_EXTENSION = ".java"
    ALL = [JAR_EXTENSION, MOJO_EXTENSION, POJO_EXTENSION]


class ArgumentsOptions(object):
    ADDRESS = "--address"
    DIR = "--dir"
    DOCKER = "--docker"
    INPUT = "--input"
    OUTPUT = "--output"
    TARGET = "--target"
    TARGET_FILENAME = "--target-csv"
    CODE_DIR = "--code-dir"
    NEGATIVE_CLASS_LABEL = "--negative-class-label"
    POSITIVE_CLASS_LABEL = "--positive-class-label"
    WEIGHTS_CSV = "--row-weights-csv"
    WEIGHTS = "--row-weights"
    SKIP_PREDICT = "--skip-predict"
    TIMEOUT = "--timeout"
    THREADED = "--threaded"
    LOGGING_LEVEL = "--logging-level"
    LANGUAGE = "--language"
    NUM_ROWS = "--num-rows"

    MAIN_COMMAND = "drum"

    SCORE = "score"
    SERVER = "server"
    FIT = "fit"
    PERF_TEST = "perf-test"
    NEW = "new"
    NEW_MODEL = "model"
    NEW_ENV = "env"
    VALIDATION = "validation"
    PUSH = "push"


class RunMode(Enum):
    SCORE = ArgumentsOptions.SCORE
    SERVER = ArgumentsOptions.SERVER
    PERF_TEST = ArgumentsOptions.PERF_TEST
    VALIDATION = ArgumentsOptions.VALIDATION
    FIT = ArgumentsOptions.FIT
    NEW = ArgumentsOptions.NEW
    PUSH = ArgumentsOptions.PUSH
    NEW_MODEL = "new_model"


class RunLanguage(Enum):
    PYTHON = "python"
    R = "r"
    JAVA = "java"


class TemplateType(object):
    MODEL = "model"
    ENV = "environment"


class EnvVarNames:
    DRUM_JAVA_XMX = "DRUM_JAVA_XMX"


@contextmanager
def verbose_stdout(verbose):
    new_target = sys.stdout
    old_target = sys.stdout
    if not verbose:
        new_target = open(os.devnull, "w")
        sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def config_logging():
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)s:  %(message)s")
