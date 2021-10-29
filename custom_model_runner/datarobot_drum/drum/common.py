"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
import sys
from typing import Optional as PythonTypingOptional

from contextlib import contextmanager
from strictyaml import Bool, Int, Map, Optional, Str, load, YAMLError, Seq, Any
from pathlib import Path

from datarobot_drum.drum.enum import (
    MODEL_CONFIG_FILENAME,
    PredictionServerMimetypes,
    TargetType,
    PayloadFormat,
)
from datarobot_drum.drum.exceptions import DrumCommonException

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
