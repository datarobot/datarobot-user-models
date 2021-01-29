import logging
import os
import sys

from contextlib import contextmanager
from enum import Enum
from strictyaml import Bool, Int, Map, Optional, Str, load, YAMLError, Seq
from pathlib import Path

DEBUG = os.environ.get("DEBUG")

LOGGER_NAME_PREFIX = "drum"
REGRESSION_PRED_COLUMN = "Predictions"
CUSTOM_FILE_NAME = "custom"
POSITIVE_CLASS_LABEL_ARG_KEYWORD = "positive_class_label"
NEGATIVE_CLASS_LABEL_ARG_KEYWORD = "negative_class_label"
CLASS_LABELS_ARG_KEYWORD = "class_labels"
TARGET_TYPE_ARG_KEYWORD = "target_type"
RESPONSE_PREDICTIONS_KEY = "predictions"
X_TRANSFORM_KEY = "X.transformed"
Y_TRANSFORM_KEY = "y.transformed"

URL_PREFIX_ENV_VAR_NAME = "URL_PREFIX"

MODEL_CONFIG_FILENAME = "model-metadata.yaml"

PERF_TEST_SERVER_LABEL = "__DRUM_PERF_TEST_SERVER__"

LOG_LEVELS = {
    "noset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class SupportedFrameworks:
    SKLEARN = "scikit-learn"
    TORCH = "torch"
    KERAS = "keras"
    XGBOOST = "xgboost"
    PYPMML = "pypmml"


extra_deps = {
    SupportedFrameworks.SKLEARN: ["scikit-learn", "scipy", "numpy"],
    SupportedFrameworks.TORCH: ["torch", "numpy", "scikit-learn", "scipy"],
    SupportedFrameworks.KERAS: ["scipy", "numpy", "h5py", "tensorflow>=2.2.1"],
    SupportedFrameworks.XGBOOST: ["scipy", "numpy", "xgboost"],
    SupportedFrameworks.PYPMML: ["pypmml"],
}


class CustomHooks:
    INIT = "init"
    READ_INPUT_DATA = "read_input_data"
    LOAD_MODEL = "load_model"
    TRANSFORM = "transform"
    SCORE = "score"
    SCORE_UNSTRUCTURED = "score_unstructured"
    POST_PROCESS = "post_process"
    FIT = "fit"

    ALL_PREDICT_STRUCTURED = [INIT, READ_INPUT_DATA, LOAD_MODEL, TRANSFORM, SCORE, POST_PROCESS]
    ALL_PREDICT_UNSTRUCTURED = [INIT, LOAD_MODEL, SCORE_UNSTRUCTURED]
    ALL_PREDICT_FIT_STRUCTURED = ALL_PREDICT_STRUCTURED + [FIT]


class UnstructuredDtoKeys:
    DATA = "data"
    QUERY = "query"
    MIMETYPE = "mimetype"
    CHARSET = "charset"


class StructuredDtoKeys:
    BINARY_DATA = "binary_data"
    MIMETYPE = "mimetype"
    TARGET_BINARY_DATA = "target_binary_data"
    TARGET_MIMETYPE = "target_mimetype"


class PredictionServerMimetypes:
    APPLICATION_JSON = "application/json"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    TEXT_PLAIN = "text/plain"
    APPLICATION_X_APACHE_ARROW_STREAM = "application/x-apache-arrow-stream"
    TEXT_MTX = "text/mtx"
    TEXT_CSV = "text/csv"
    EMPTY = ""


class InputFormatExtension:
    MTX = ".mtx"
    ARROW = ".arrow"


InputFormatToMimetype = {
    InputFormatExtension.MTX: PredictionServerMimetypes.TEXT_MTX,
    InputFormatExtension.ARROW: PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM,
}


class PythonArtifacts:
    PKL_EXTENSION = ".pkl"
    TORCH_EXTENSION = ".pth"
    KERAS_EXTENSION = ".h5"
    JOBLIB_EXTENSION = ".joblib"
    PYPMML_EXTENSION = ".pmml"
    ALL = [PKL_EXTENSION, TORCH_EXTENSION, KERAS_EXTENSION, JOBLIB_EXTENSION, PYPMML_EXTENSION]


class RArtifacts:
    RDS_EXTENSION = ".rds"
    ALL = [RDS_EXTENSION]


class JavaArtifacts:
    JAR_EXTENSION = ".jar"
    MOJO_EXTENSION = ".zip"
    POJO_EXTENSION = ".java"
    MOJO_PIPELINE_EXTENSION = ".mojo"
    ALL = [JAR_EXTENSION, MOJO_EXTENSION, POJO_EXTENSION, MOJO_PIPELINE_EXTENSION]


class ArgumentsOptions:
    ADDRESS = "--address"
    DIR = "--dir"
    DOCKER = "--docker"
    MEMORY = "--memory"
    INPUT = "--input"
    OUTPUT = "--output"
    TARGET = "--target"
    TARGET_CSV = "--target-csv"
    CODE_DIR = "--code-dir"
    NEGATIVE_CLASS_LABEL = "--negative-class-label"
    POSITIVE_CLASS_LABEL = "--positive-class-label"
    WEIGHTS_CSV = "--row-weights-csv"
    WEIGHTS = "--row-weights"
    SKIP_PREDICT = "--skip-predict"
    TIMEOUT = "--timeout"
    PRODUCTION = "--production"
    LOGGING_LEVEL = "--logging-level"
    LANGUAGE = "--language"
    NUM_ROWS = "--num-rows"
    MONITOR = "--monitor"
    DEPLOYMENT_ID = "--deployment-id"
    MODEL_ID = "--model-id"
    MONITOR_SETTINGS = "--monitor-settings"
    QUERY = "--query"
    CONTENT_TYPE = "--content-type"
    WITH_ERROR_SERVER = "--with-error-server"
    SHOW_STACKTRACE = "--show-stacktrace"
    MAX_WORKERS = "--max-workers"
    VERBOSE = "--verbose"
    VERSION = "--version"
    TARGET_TYPE = "--target-type"
    CLASS_LABELS = "--class-labels"
    CLASS_LABELS_FILE = "--class-labels-file"

    MAIN_COMMAND = "drum" if not DEBUG else "./custom_model_runner/bin/drum"

    SCORE = "score"
    SERVER = "server"
    FIT = "fit"
    PERF_TEST = "perf-test"
    NEW = "new"
    NEW_MODEL = "model"
    NEW_ENV = "env"
    VALIDATION = "validation"
    PUSH = "push"


class ArgumentOptionsEnvVars:
    TARGET_TYPE = "TARGET_TYPE"
    CODE_DIR = "CODE_DIR"
    NEGATIVE_CLASS_LABEL = "NEGATIVE_CLASS_LABEL"
    POSITIVE_CLASS_LABEL = "POSITIVE_CLASS_LABEL"
    CLASS_LABELS_FILE = "CLASS_LABELS_FILE"
    CLASS_LABELS = "CLASS_LABELS"
    ADDRESS = "ADDRESS"
    MAX_WORKERS = "MAX_WORKERS"

    MONITOR = "MONITOR"
    WITH_ERROR_SERVER = "WITH_ERROR_SERVER"
    SHOW_STACKTRACE = "SHOW_STACKTRACE"
    PRODUCTION = "PRODUCTION"

    SKIP_PREDICT = "SKIP_PREDICT"

    VALUE_VARS = [
        TARGET_TYPE,
        CODE_DIR,
        NEGATIVE_CLASS_LABEL,
        POSITIVE_CLASS_LABEL,
        CLASS_LABELS_FILE,
        CLASS_LABELS,
        ADDRESS,
        MAX_WORKERS,
    ]
    BOOL_VARS = [WITH_ERROR_SERVER, SHOW_STACKTRACE, PRODUCTION, MONITOR, SKIP_PREDICT]


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


class TargetType(Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    UNSTRUCTURED = "unstructured"
    MULTICLASS = "multiclass"
    TRANSFORM = "transform"
    CLASSIFICATION = [BINARY, MULTICLASS]
    ALL = [BINARY, MULTICLASS, REGRESSION, ANOMALY, UNSTRUCTURED, TRANSFORM]


class TemplateType:
    MODEL = "model"
    ENV = "environment"


class EnvVarNames:
    DRUM_JAVA_XMX = "DRUM_JAVA_XMX"
    DRUM_JAVA_CUSTOM_PREDICTOR_CLASS = "DRUM_JAVA_CUSTOM_PREDICTOR_CLASS"
    DRUM_JAVA_CUSTOM_CLASS_PATH = "DRUM_JAVA_CUSTOM_CLASS_PATH"


@contextmanager
def reroute_stdout_to_stderr():
    keep = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = keep


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


MODEL_CONFIG_SCHEMA = Map(
    {
        "name": Str(),
        "type": Str(),
        "environmentID": Str(),
        "targetType": Str(),
        "validation": Map({"input": Str(), Optional("targetName"): Str()}),
        Optional("modelID"): Str(),
        Optional("description"): Str(),
        Optional("majorVersion"): Bool(),
        Optional("inferenceModel"): Map(
            {
                "targetName": Str(),
                Optional("positiveClassLabel"): Str(),
                Optional("negativeClassLabel"): Str(),
                Optional("classLabels"): Seq(Str()),
                Optional("classLabelsFile"): Str(),
                Optional("predictionThreshold"): Int(),
            }
        ),
        Optional("trainingModel"): Map({Optional("trainOnProject"): Str()}),
    }
)


def read_model_metadata_yaml(code_dir):
    code_dir = Path(code_dir)
    config_path = code_dir.joinpath(MODEL_CONFIG_FILENAME)
    if config_path.exists():
        with open(config_path) as f:
            try:
                model_config = load(f.read(), MODEL_CONFIG_SCHEMA).data
            except YAMLError as e:
                print(e)
                raise SystemExit(1)
        return model_config
    return None


class PayloadFormat:
    CSV = "csv"
    ARROW = "arrow"
    MTX = "mtx"


class SupportedPayloadFormats:
    def __init__(self):
        self._formats = {}
        self._mimetype_to_payload_format = {
            None: PayloadFormat.CSV,
            PredictionServerMimetypes.EMPTY: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_CSV: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_PLAIN: PayloadFormat.CSV,
            PredictionServerMimetypes.TEXT_MTX: PayloadFormat.MTX,
            PredictionServerMimetypes.APPLICATION_X_APACHE_ARROW_STREAM: PayloadFormat.ARROW,
        }

    def add(self, payload_format, format_version=None):
        self._formats[payload_format] = format_version

    def is_mimetype_supported(self, mimetype):
        payload_format = self._mimetype_to_payload_format.get(mimetype)
        if payload_format is None:
            return False

        return payload_format in self._formats

    def __iter__(self):
        for payload_format, format_version in self._formats.items():
            yield payload_format, format_version


def make_predictor_capabilities(supported_payload_formats):
    return {
        "supported_payload_formats": {
            payload_format: format_version
            for payload_format, format_version in supported_payload_formats
        }
    }


try:
    import pyarrow
except ImportError:
    pyarrow = None


def get_pyarrow_module():
    return pyarrow


def verify_pyarrow_module():
    if pyarrow is None:
        raise ModuleNotFoundError("Please install pyarrow to support Arrow format")
    return pyarrow
