"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import logging
import os
import re
import sys
from distutils.util import strtobool
from typing import Dict, List
from typing import Optional as PythonTypingOptional

from contextlib import contextmanager
from strictyaml import Bool, Int, Map, Optional, Str, load, YAMLError, Seq, Any, YAMLValidationError
from pathlib import Path
import six
import trafaret as t

from datarobot_drum.drum.enum import (
    MODEL_CONFIG_FILENAME,
    ModelMetadataKeys,
    ModelMetadataHyperParamTypes,
    PredictionServerMimetypes,
    TargetType,
    PayloadFormat,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.model_metadata import HyperParameterTrafaret

from datarobot_drum.drum.typeschema_validation import (
    get_type_schema_yaml_validator,
    revalidate_typeschema,
)


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


def get_metadata(options):
    code_dir = Path(options.code_dir)
    if options.model_config is None:
        raise DrumCommonException(
            "You must have a file with the name {} in the directory {}. \n"
            "You don't. \nWhat you do have is these files: \n{} ".format(
                MODEL_CONFIG_FILENAME, code_dir, os.listdir(code_dir)
            )
        )
    return options.model_config


MODEL_CONFIG_SCHEMA = Map(
    {
        ModelMetadataKeys.NAME: Str(),
        ModelMetadataKeys.TYPE: Str(),
        ModelMetadataKeys.TARGET_TYPE: Str(),
        Optional(ModelMetadataKeys.ENVIRONMENT_ID): Str(),
        Optional(ModelMetadataKeys.VALIDATION): Map(
            {"input": Str(), Optional("targetName"): Str()}
        ),
        Optional(ModelMetadataKeys.MODEL_ID): Str(),
        Optional(ModelMetadataKeys.DESCRIPTION): Str(),
        Optional(ModelMetadataKeys.MAJOR_VERSION): Bool(),
        Optional(ModelMetadataKeys.INFERENCE_MODEL): Map(
            {
                Optional("targetName"): Str(),
                Optional("positiveClassLabel"): Str(),
                Optional("negativeClassLabel"): Str(),
                Optional("classLabels"): Seq(Str()),
                Optional("classLabelsFile"): Str(),
                Optional("predictionThreshold"): Int(),
            }
        ),
        Optional(ModelMetadataKeys.TRAINING_MODEL): Map({Optional("trainOnProject"): Str()}),
        Optional(ModelMetadataKeys.HYPERPARAMETERS): Any(),
        Optional(ModelMetadataKeys.VALIDATION_SCHEMA): get_type_schema_yaml_validator(),
        Optional(ModelMetadataKeys.CUSTOM_PREDICTOR): Any(),
    }
)


def validate_config_fields(model_config, *fields):
    missing_sections = []
    for f in fields:
        if f not in model_config:
            missing_sections.append(f)

    if missing_sections:
        raise DrumCommonException(
            "The following keys are missing in {} file.\n"
            "Missing keys: {}".format(MODEL_CONFIG_FILENAME, missing_sections)
        )


def validate_model_metadata_hyperparameter(hyper_params: List) -> None:
    """Validate hyperparameters section in model metadata yaml.

    Parameters
    ----------
    hyper_params:
        A list of hyperparameter definitions.

    Raises
    ------
    DrumCommonException
        Raised when validation fails.
    """

    def _validate_param_name(param_name: str):
        allowed_char_re = "^([a-zA-Z]+[a-zA-Z_]*[a-zA-Z]+|[a-zA-Z]+)$"

        if re.match(allowed_char_re, param_name) is None:
            error_msg = (
                "Invalid param name: {param_name}. "
                "Only Eng characters and underscore are allowed. The parameter name should not start or end with the "
                "underscore."
            )
            error_msg = error_msg.format(param_name=param_name)
            raise DrumCommonException(error_msg)

    def _validate_numeric_parameter(param: Dict):
        param_name = param["name"]
        param_type = param["type"]
        min_val = param["min"]
        max_val = param["max"]
        default_val = param.get("default")
        if min_val >= max_val:
            error_msg = "Invalid {} parameter {}: max must be greater than min".format(
                param_type, param_name
            )
            raise DrumCommonException(error_msg)
        if default_val is not None:
            if default_val > max_val or default_val < min_val:
                error_msg = "Invalid {} parameter {}: values must be between [{}, {}]".format(
                    param_type, param_name, min_val, max_val
                )
                raise DrumCommonException(error_msg)

    def _validate_multi_parameter(multi_params: Dict):
        param_name = multi_params["name"]
        multi_params = multi_params["values"]
        for param_type, param in six.iteritems(multi_params):
            if param_type in {"int", "float"}:
                _param = dict(
                    {"name": "{}__{}".format(param_name, param_type), "type": param_type}, **param
                )
                _validate_numeric_parameter(_param)

    try:
        for param in hyper_params:
            param_type = param.get("type")
            if not param_type:
                raise DrumCommonException('"type": "is required"')
            if param_type not in ModelMetadataHyperParamTypes.all():
                raise DrumCommonException({"type": "is invalid"})
            param = HyperParameterTrafaret[param_type].transform(param)
            if param_type == ModelMetadataHyperParamTypes.INT:
                _validate_numeric_parameter(param)
            elif param_type == ModelMetadataHyperParamTypes.FLOAT:
                _validate_numeric_parameter(param)
            elif param_type == ModelMetadataHyperParamTypes.MULTI:
                _validate_multi_parameter(param)
            _validate_param_name(param["name"])
    except t.DataError as e:
        raise DrumCommonException(json.dumps(e.as_dict(value=True)))


def read_model_metadata_yaml(code_dir) -> PythonTypingOptional[dict]:
    code_dir = Path(code_dir)
    config_path = code_dir.joinpath(MODEL_CONFIG_FILENAME)
    if config_path.exists():
        with open(config_path) as f:
            try:
                model_config = load(f.read(), MODEL_CONFIG_SCHEMA)
                if "typeSchema" in model_config:
                    revalidate_typeschema(model_config["typeSchema"])
                model_config = model_config.data
            except YAMLError as e:
                if "found a blank string" in e.problem:
                    print("The model_metadata.yaml file appears to be empty.")
                else:
                    print(e)
                raise SystemExit(1)

        if model_config[ModelMetadataKeys.TARGET_TYPE] == TargetType.BINARY.value:
            if model_config[ModelMetadataKeys.TYPE] == "inference":
                validate_config_fields(model_config, ModelMetadataKeys.INFERENCE_MODEL)
                validate_config_fields(
                    model_config[ModelMetadataKeys.INFERENCE_MODEL],
                    *["positiveClassLabel", "negativeClassLabel"]
                )

        if model_config[ModelMetadataKeys.TARGET_TYPE] == TargetType.MULTICLASS.value:
            if model_config[ModelMetadataKeys.TYPE] == "inference":
                validate_config_fields(model_config, ModelMetadataKeys.INFERENCE_MODEL)
                classLabelsKeyIn = "classLabels" in model_config[ModelMetadataKeys.INFERENCE_MODEL]
                classLabelFileKeyIn = (
                    "classLabelsFile" in model_config[ModelMetadataKeys.INFERENCE_MODEL]
                )
                if all([classLabelsKeyIn, classLabelFileKeyIn]):
                    raise DrumCommonException(
                        "\nError - for multiclass classification, either the class labels or "
                        "a class labels file should be provided in {} file, but not both.".format(
                            MODEL_CONFIG_FILENAME
                        )
                    )
                elif not any([classLabelsKeyIn, classLabelFileKeyIn]):
                    raise DrumCommonException(
                        "\nError - for multiclass classification, either the class labels or "
                        "a class labels file must be provided in {} file.".format(
                            MODEL_CONFIG_FILENAME
                        )
                    )

                if classLabelFileKeyIn:
                    classLabelsFile = model_config[ModelMetadataKeys.INFERENCE_MODEL][
                        "classLabelsFile"
                    ]

                    with open(classLabelsFile) as f:
                        labels = [label for label in f.read().split(os.linesep) if label]
                        if len(labels) < 2:
                            raise DrumCommonException(
                                "Multiclass classification requires at least 2 labels."
                            )
                        model_config[ModelMetadataKeys.INFERENCE_MODEL]["classLabels"] = labels
                        model_config[ModelMetadataKeys.INFERENCE_MODEL]["classLabelsFile"] = None

        hyper_params = model_config.get(ModelMetadataKeys.HYPERPARAMETERS)
        if hyper_params:
            validate_model_metadata_hyperparameter(hyper_params)

        return model_config
    return None


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


def to_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return strtobool(value)


FIT_METADATA_FILENAME = "fit_runtime_data.json"
