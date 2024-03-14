"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import re
import six
import trafaret as t

from pathlib import Path
from ruamel.yaml import YAMLError
from strictyaml import (
    load,
    Map,
    Str,
    Optional,
    Bool,
    Seq,
    Int,
    Any,
    StrictYAMLError,
    YAMLValidationError,
)
from typing import Optional as PythonTypingOptional, List, Dict

from datarobot_drum.drum.enum import (
    ModelMetadataHyperParamTypes,
    MODEL_CONFIG_FILENAME,
    ModelMetadataKeys,
    TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException, DrumFormatSchemaException
from datarobot_drum.drum.typeschema_validation import (
    revalidate_typeschema,
    get_type_schema_yaml_validator,
)


# Max length of a user-defined parameter
PARAM_NAME_MAX_LENGTH = 64

# Max length of a select value
PARAM_SELECT_VALUE_MAX_LENGTH = 32

# Max number of possible select values
PARAM_SELECT_NUM_VALUES_MAX_LENGTH = 24

# Max length of the value of a string/unicode parameter
PARAM_STRING_MAX_LENGTH = 1024


class ParamNameTrafaret(t.String):
    def __init__(self, *args, **kw_args):
        super(ParamNameTrafaret, self).__init__(*args, max_length=PARAM_NAME_MAX_LENGTH, **kw_args)

    def check_and_return(self, value):
        try:
            return super(ParamNameTrafaret, self).check_and_return(value)
        except t.DataError as e:
            error_msg = "Invalid parameter name: {}".format(str(e))
            raise t.DataError(error_msg)


IntHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("int"),
        t.Key("min"): t.ToInt,
        t.Key("max"): t.ToInt,
        t.Key("default", optional=True): t.ToInt,
    }
)

FloatHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("float"),
        t.Key("min"): t.ToFloat,
        t.Key("max"): t.ToFloat,
        t.Key("default", optional=True): t.ToFloat,
    }
)

StringHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("string"),
        t.Key("default", optional=True): t.String(
            max_length=PARAM_STRING_MAX_LENGTH, allow_blank=True
        ),
    }
)

SelectHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("select"),
        t.Key("values"): t.List(
            t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH, allow_blank=False),
            min_length=1,
            max_length=PARAM_SELECT_NUM_VALUES_MAX_LENGTH,
        ),
        t.Key("default", optional=True): t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH),
    }
)

# Multi only supports int, float, or select
MultiHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("multi"),
        t.Key("values"): t.Dict(
            {
                t.Key("int", optional=True): t.Dict(
                    {
                        t.Key("min"): t.ToInt,
                        t.Key("max"): t.ToInt,
                    }
                ),
                t.Key("float", optional=True): t.Dict(
                    {
                        t.Key("min"): t.ToFloat,
                        t.Key("max"): t.ToFloat,
                    }
                ),
                t.Key("select", optional=True): t.Dict(
                    {
                        t.Key("values"): t.List(
                            t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH, allow_blank=False),
                            min_length=1,
                            max_length=PARAM_SELECT_NUM_VALUES_MAX_LENGTH,
                        ),
                    }
                ),
            }
        ),
        t.Key("default", optional=True): t.Or(
            t.Int, t.Float, t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH)
        ),
    }
)


HyperParameterTrafaret = {
    ModelMetadataHyperParamTypes.INT: IntHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.FLOAT: FloatHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.STRING: StringHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.SELECT: SelectHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.MULTI: MultiHyperParameterTrafaret,
}


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
            except YAMLValidationError as e:
                if "found a blank string" in e.problem:
                    print("The model_metadata.yaml file appears to be empty.")
                else:
                    print(e)
                raise SystemExit(1)
            except StrictYAMLError as e:
                raise DrumFormatSchemaException(
                    "\nStrictYAMLError: The current format does not comply with strict yaml rules."
                    " (Empty list on fields are not allowed)\n{}".format(e)
                )
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

        hyper_params = model_config.get(ModelMetadataKeys.HYPERPARAMETERS)
        if hyper_params:
            validate_model_metadata_hyperparameter(hyper_params)

        return model_config
    return None


def read_default_model_metadata_yaml() -> PythonTypingOptional[dict]:
    default_type_schema_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "resource", "default_typeschema")
    )
    return read_model_metadata_yaml(default_type_schema_path)["typeSchema"]


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
        Optional(ModelMetadataKeys.RUNTIME_PARAMETERS): Any(),
        Optional(ModelMetadataKeys.USER_CREDENTIAL_SPECIFICATIONS): Seq(
            Map({"key": Str(), "valueFrom": Str(), Optional("reminder"): Str()})
        ),
    }
)
