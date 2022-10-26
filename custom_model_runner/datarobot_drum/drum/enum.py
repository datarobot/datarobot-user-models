"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
from enum import Enum


DEBUG = os.environ.get("DEBUG")


LOGGER_NAME_PREFIX = "drum"


REGRESSION_PRED_COLUMN = "Predictions"


CUSTOM_FILE_NAME = "custom"


FLASK_EXT_FILE_NAME = "custom_flask"


POSITIVE_CLASS_LABEL_ARG_KEYWORD = "positive_class_label"


NEGATIVE_CLASS_LABEL_ARG_KEYWORD = "negative_class_label"


CLASS_LABELS_ARG_KEYWORD = "class_labels"


TARGET_TYPE_ARG_KEYWORD = "target_type"


RESPONSE_PREDICTIONS_KEY = "predictions"


X_FORMAT_KEY = "X.format"


X_TRANSFORM_KEY = "X.transformed"


Y_TRANSFORM_KEY = "y.transformed"


SPARSE_COLNAMES = "X.colnames"


URL_PREFIX_ENV_VAR_NAME = "URL_PREFIX"


MODEL_CONFIG_FILENAME = "model-metadata.yaml"


PERF_TEST_SERVER_LABEL = "__DRUM_PERF_TEST_SERVER__"

CUSTOM_PY_CLASS_NAME = "CustomTask"

LOG_LEVELS = {
    "all": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.CRITICAL,
}


class SupportedFrameworks:
    SKLEARN = "scikit-learn"
    TORCH = "torch"
    KERAS = "keras"
    XGBOOST = "xgboost"
    PYPMML = "pypmml"
    ONNX = "onnx"


extra_deps = {
    SupportedFrameworks.SKLEARN: ["scikit-learn", "scipy", "numpy"],
    SupportedFrameworks.TORCH: ["torch", "numpy", "scikit-learn", "scipy"],
    SupportedFrameworks.KERAS: ["scipy", "numpy", "h5py", "tensorflow>=2.2.1"],
    SupportedFrameworks.XGBOOST: ["scipy", "numpy", "xgboost"],
    SupportedFrameworks.PYPMML: ["pypmml"],
    SupportedFrameworks.ONNX: ["onnxruntime"],
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
    SPARSE_COLNAMES = "sparse_colnames"
    PARAMETERS = "parameters"


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
    CSV = ".csv"


class ModelInfoKeys:
    CODE_DIR = "codeDir"
    TARGET_TYPE = "targetType"
    PREDICTOR = "predictor"
    LANGUAGE = "language"
    DRUM_VERSION = "drumVersion"
    DRUM_SERVER = "drumServer"
    MODEL_METADATA = "modelMetadata"
    POSITIVE_CLASS_LABEL = "positiveClassLabel"
    NEGATIVE_CLASS_LABEL = "negativeClassLabel"
    CLASS_LABELS = "classLabels"

    REQUIRED = [CODE_DIR, TARGET_TYPE, LANGUAGE, DRUM_VERSION, DRUM_SERVER]


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
    ONNX_EXTENSION = ".onnx"
    ALL = [
        PKL_EXTENSION,
        TORCH_EXTENSION,
        KERAS_EXTENSION,
        JOBLIB_EXTENSION,
        PYPMML_EXTENSION,
        ONNX_EXTENSION,
    ]


class RArtifacts:
    RDS_EXTENSION = ".rds"
    ALL = [RDS_EXTENSION]


class JavaArtifacts:
    JAR_EXTENSION = ".jar"
    MOJO_EXTENSION = ".zip"
    POJO_EXTENSION = ".java"
    MOJO_PIPELINE_EXTENSION = ".mojo"
    ALL = [JAR_EXTENSION, MOJO_EXTENSION, POJO_EXTENSION, MOJO_PIPELINE_EXTENSION]


class JuliaArtifacts:
    JLSO_EXTENSION = ".jlso"
    ALL = [JLSO_EXTENSION]


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
    MONITOR_EMBEDDED = "--monitor-embedded"
    DEPLOYMENT_ID = "--deployment-id"
    MODEL_ID = "--model-id"
    MONITOR_SETTINGS = "--monitor-settings"
    DR_WEBSERVER = "--webserver"
    DR_API_TOKEN = "--api-token"
    DEPLOYMENT_CONFIG = "--deployment-config"
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
    SKIP_DEPS_INSTALL = "--skip-deps-install"
    SPARSE_COLFILE = "--sparse-column-file"
    PARAMETER_FILE = "--parameter-file"
    DISABLE_STRICT_VALIDATION = "--disable-strict-validation"
    ENABLE_PREDICT_METRICS_REPORT = "--enable-fit-metadata"
    ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS = "--allow-dr-api-access"

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
    DEPLOYMENT_CONFIG = "DEPLOYMENT_CONFIG"

    MONITOR = "MONITOR"
    MONITOR_EMBEDDED = "MLOPS_REPORTING_FROM_UNSTRUCTURED_MODELS"
    WITH_ERROR_SERVER = "WITH_ERROR_SERVER"
    SHOW_STACKTRACE = "SHOW_STACKTRACE"
    PRODUCTION = "PRODUCTION"

    SKIP_PREDICT = "SKIP_PREDICT"

    ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS = "ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"

    VALUE_VARS = [
        TARGET_TYPE,
        CODE_DIR,
        NEGATIVE_CLASS_LABEL,
        POSITIVE_CLASS_LABEL,
        CLASS_LABELS_FILE,
        CLASS_LABELS,
        ADDRESS,
        MAX_WORKERS,
        DEPLOYMENT_CONFIG,
    ]
    BOOL_VARS = [
        WITH_ERROR_SERVER,
        SHOW_STACKTRACE,
        PRODUCTION,
        MONITOR,
        MONITOR_EMBEDDED,
        SKIP_PREDICT,
        ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS,
    ]

    @classmethod
    def to_arg_option(cls, env_var):
        if env_var == cls.MONITOR_EMBEDDED:
            return ArgumentsOptions.MONITOR_EMBEDDED
        return ArgumentsOptions.__dict__[env_var]


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
    JULIA = "julia"


class TargetType(Enum):
    # Update documentation in model-metadata.md if a new type is added here.
    BINARY = "binary"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    UNSTRUCTURED = "unstructured"
    MULTICLASS = "multiclass"
    TRANSFORM = "transform"
    CLASSIFICATION = [BINARY, MULTICLASS]
    SINGLE_COL = [REGRESSION, ANOMALY]
    ALL = [BINARY, MULTICLASS, REGRESSION, ANOMALY, UNSTRUCTURED, TRANSFORM]


class TemplateType:
    MODEL = "model"
    ENV = "environment"


class EnvVarNames:
    DRUM_JAVA_XMX = "DRUM_JAVA_XMX"
    DRUM_JAVA_CUSTOM_PREDICTOR_CLASS = "DRUM_JAVA_CUSTOM_PREDICTOR_CLASS"
    DRUM_JAVA_CUSTOM_CLASS_PATH = "DRUM_JAVA_CUSTOM_CLASS_PATH"


class PayloadFormat:
    CSV = "csv"
    ARROW = "arrow"
    MTX = "mtx"


class ExitCodes(Enum):
    # This is the DRUM specific exit code. Please avoid using reserved/common exit codes. e.g.,
    # 1, 2, 126, 127, 128, 128+n, 130, 225*
    SCHEMA_VALIDATION_ERROR = 3  # used when the program exits due to custom task validation fails.


class ModelMetadataKeys(object):
    NAME = "name"
    TYPE = "type"
    TARGET_TYPE = "targetType"
    ENVIRONMENT_ID = "environmentID"
    VALIDATION = "validation"
    MODEL_ID = "modelID"
    DESCRIPTION = "description"
    MAJOR_VERSION = "majorVersion"
    INFERENCE_MODEL = "inferenceModel"
    TRAINING_MODEL = "trainingModel"
    HYPERPARAMETERS = "hyperparameters"
    VALIDATION_SCHEMA = "typeSchema"
    # customPredictor section is not used by DRUM,
    # it is a place holder if user wants to add some fields and read them on his own
    CUSTOM_PREDICTOR = "customPredictor"


class ModelMetadataHyperParamTypes(object):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    SELECT = "select"
    MULTI = "multi"

    @classmethod
    def all(cls):
        return {cls.INT, cls.FLOAT, cls.STRING, cls.SELECT, cls.MULTI}


class ModelMetadataMultiHyperParamTypes(object):
    INT = ModelMetadataHyperParamTypes.INT
    FLOAT = ModelMetadataHyperParamTypes.FLOAT
    SELECT = ModelMetadataHyperParamTypes.SELECT

    @classmethod
    def all(cls):
        return set(cls.all_list())

    @classmethod
    def all_list(cls):
        return [cls.INT, cls.FLOAT, cls.SELECT]
