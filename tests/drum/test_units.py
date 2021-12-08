# -*- coding: utf-8 -*-
"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import copy
import glob
import itertools
import json
import os
import socket
import tempfile
import time
from argparse import Namespace
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
import yaml

import numpy as np
import pandas as pd
import pyarrow
import pytest
import responses
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from datarobot_drum.drum.artifact_predictors.sklearn_predictor import SKLearnPredictor
from datarobot_drum.drum.artifact_predictors.pmml_predictor import PMMLPredictor
from datarobot_drum.drum.artifact_predictors.xgboost_predictor import XGBoostPredictor
from datarobot_drum.drum.artifact_predictors.keras_predictor import KerasPredictor
from datarobot_drum.drum.artifact_predictors.torch_predictor import PyTorchPredictor
from datarobot_drum.drum.common import (
    read_model_metadata_yaml,
    validate_config_fields,
)
from datarobot_drum.drum.enum import (
    MODEL_CONFIG_FILENAME,
    TargetType,
    RunLanguage,
    RunMode,
    ModelMetadataKeys,
    ModelMetadataMultiHyperParamTypes,
)
from datarobot_drum.drum.drum import (
    CMRunner,
    create_custom_inference_model_folder,
    output_in_code_dir,
    possibly_intuit_order,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import JavaPredictor
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from datarobot_drum.drum.language_predictors.r_predictor.r_predictor import RPredictor
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.model_metadata import PARAM_NAME_MAX_LENGTH
from datarobot_drum.drum.model_metadata import PARAM_SELECT_NUM_VALUES_MAX_LENGTH
from datarobot_drum.drum.model_metadata import PARAM_SELECT_VALUE_MAX_LENGTH
from datarobot_drum.drum.model_metadata import PARAM_STRING_MAX_LENGTH
from datarobot_drum.drum.push import _push_inference, _push_training, drum_push
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.drum.utils import DrumUtils
from datarobot_drum.drum.data_marshalling import _marshal_labels
from datarobot_drum.drum.typeschema_validation import SchemaValidator
from datarobot_drum.drum.utils import StructuredInputReadUtils
from tests.drum.constants import TESTS_DATA_PATH


class TestOrderIntuition:
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
    binary_filename = os.path.join(tests_data_path, "iris_binary_training.csv")
    regression_filename = os.path.join(tests_data_path, "juniors_3_year_stats_regression.csv")
    one_target_filename = os.path.join(tests_data_path, "one_target.csv")

    def test_colname(self):
        classes = possibly_intuit_order(self.binary_filename, target_col_name="Species")
        assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_colfile(self):
        with NamedTemporaryFile() as target_file:
            df = pd.read_csv(self.binary_filename)
            with open(target_file.name, "w") as f:
                target_series = df["Species"]
                target_series.to_csv(f, index=False, header="Target")

            classes = possibly_intuit_order(self.binary_filename, target_data_file=target_file.name)
            assert set(classes) == {"Iris-versicolor", "Iris-setosa"}

    def test_badfile(self):
        with pytest.raises(DrumCommonException):
            possibly_intuit_order(self.one_target_filename, target_col_name="Species")

    def test_unsupervised(self):
        classes = possibly_intuit_order(
            self.regression_filename, target_col_name="Grade 2014", is_anomaly=True
        )
        assert classes is None


modelID = "5f1f15a4d6111f01cb7f91f"
environmentID = "5e8c889607389fe0f466c72d"
projectID = "abc123"


@pytest.fixture
def basic_model_metadata_yaml():
    return {
        "name": "basic-model-metadata-yaml",
        "type": "inference",
        "targetType": "regression",
    }


@pytest.fixture
def complete_int_hyper_param():
    return {
        "name": "param_int",
        "type": "int",
        "min": 0,
        "max": 1,
        "default": 0,
    }


@pytest.fixture
def complete_float_hyper_param():
    return {
        "name": "param_float",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
    }


@pytest.fixture
def complete_string_hyper_param():
    return {
        "name": "param_string",
        "type": "string",
        "default": "default string",
    }


@pytest.fixture
def complete_select_hyper_param():
    return {
        "name": "param_select",
        "type": "select",
        "values": ["value 1", "value 2"],
        "default": "value 1",
    }


@pytest.fixture
def complete_multi_hyper_param():
    return {
        "name": "param_multi",
        "type": "multi",
        "values": {
            "int": {"min": 0, "max": 1},
            "float": {"min": 0.0, "max": 1.0},
            "select": {"values": ["value 1", "value 2"]},
        },
        "default": "value 1",
    }


@pytest.fixture
def inference_metadata_yaml():
    return dedent(
        """
        name: drumpush-regression
        type: inference
        targetType: regression
        environmentID: {environmentID}
        inferenceModel:
          targetName: Grade 2014
        validation:
          input: hello
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def inference_binary_metadata_yaml_no_target_name():
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        environmentID: {environmentID}
        inferenceModel:
          positiveClassLabel: yes
          negativeClassLabel: no
        validation:
          input: hello
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def inference_binary_metadata_no_label():
    return dedent(
        """
        name: drumpush-binary
        type: inference
        targetType: binary
        inferenceModel:
          positiveClassLabel: yes
        """
    )


@pytest.fixture
def multiclass_labels():
    return ["GALAXY", "QSO", "STAR"]


@pytest.fixture
def inference_multiclass_metadata_yaml_no_labels():
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
        validation:
          input: hello
        """
    ).format(environmentID)


@pytest.fixture
def inference_multiclass_metadata_yaml(multiclass_labels):
    return dedent(
        """
        name: drumpush-multiclass
        type: inference
        targetType: multiclass
        environmentID: {}
        inferenceModel:
          targetName: class
          classLabels:
            - {}
            - {}
            - {}
        validation:
          input: hello
        """
    ).format(environmentID, *multiclass_labels)


@pytest.fixture
def inference_multiclass_metadata_yaml_label_file(multiclass_labels):
    with NamedTemporaryFile(mode="w+") as f:
        f.write("\n".join(multiclass_labels))
        f.flush()
        yield dedent(
            """
            name: drumpush-multiclass
            type: inference
            targetType: multiclass
            environmentID: {}
            inferenceModel:
              targetName: class
              classLabelsFile: {}
            validation:
              input: hello
            """
        ).format(environmentID, f.name)


@pytest.fixture
def inference_multiclass_metadata_yaml_labels_and_label_file(multiclass_labels):
    with NamedTemporaryFile(mode="w+") as f:
        f.write("\n".join(multiclass_labels))
        f.flush()
        yield dedent(
            """
            name: drumpush-multiclass
            type: inference
            targetType: multiclass
            environmentID: {}
            inferenceModel:
              targetName: class
              classLabelsFile: {}
              classLabels:
                - {}
                - {}
                - {}
            validation:
              input: hello
            """
        ).format(environmentID, f.name, *multiclass_labels)


@pytest.fixture
def training_metadata_yaml():
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        validation:
           input: hello 
        """
    ).format(environmentID=environmentID)


@pytest.fixture
def training_metadata_yaml_with_proj():
    return dedent(
        """
        name: drumpush-regression
        type: training
        targetType: regression
        environmentID: {environmentID}
        trainingModel:
            trainOnProject: {projectID}
        validation:
            input: hello 
        """
    ).format(environmentID=environmentID, projectID=projectID)


@pytest.fixture
def custom_predictor_metadata_yaml():
    return dedent(
        """
        name: model-with-custom-java-predictor
        type: inference
        targetType: regression
        customPredictor:
           arbitraryField: This info is read directly by a custom predictor
        """
    )


version_response = {
    "id": "1",
    "custom_model_id": "1",
    "version_minor": 1,
    "version_major": 1,
    "is_frozen": False,
    "items": [{"id": "1", "file_name": "hi", "file_path": "hi", "file_source": "hi"}],
}

tasks_version_response = {
    "id": "1",
    "custom_task_id": "1",
    "version_minor": 1,
    "version_major": 1,
    "is_frozen": False,
    "items": [
        {
            "id": "1",
            "file_name": "hi",
            "file_path": "hi",
            "file_source": "hi",
            "created": str(time.time()),
        }
    ],
    "label": "test",
    "created": str(time.time()),
}


@pytest.mark.parametrize(
    "config_yaml",
    [
        "custom_predictor_metadata_yaml",
        "training_metadata_yaml",
        "training_metadata_yaml_with_proj",
        "inference_metadata_yaml",
        "inference_multiclass_metadata_yaml",
        "inference_multiclass_metadata_yaml_label_file",
    ],
)
@pytest.mark.parametrize("existing_model_id", [None])
def test_yaml_metadata(request, config_yaml, existing_model_id, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    if existing_model_id:
        config_yaml = config_yaml + "\nmodelID: {}".format(existing_model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize(
    "config_yaml, test_case_number",
    [
        ("custom_predictor_metadata_yaml", 1),
        ("inference_binary_metadata_no_label", 2),
        ("inference_multiclass_metadata_yaml_no_labels", 3),
        ("inference_multiclass_metadata_yaml_labels_and_label_file", 4),
        ("inference_multiclass_metadata_yaml", 100),
        ("inference_multiclass_metadata_yaml_label_file", 100),
    ],
)
def test_yaml_metadata_missing_fields(tmp_path, config_yaml, request, test_case_number):
    config_yaml = request.getfixturevalue(config_yaml)
    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)

    if test_case_number == 1:
        conf = read_model_metadata_yaml(tmp_path)
        with pytest.raises(
            DrumCommonException, match="Missing keys: \['validation', 'environmentID'\]"
        ):
            validate_config_fields(
                conf,
                ModelMetadataKeys.CUSTOM_PREDICTOR,
                ModelMetadataKeys.VALIDATION,
                ModelMetadataKeys.ENVIRONMENT_ID,
            )
    elif test_case_number == 2:
        with pytest.raises(DrumCommonException, match="Missing keys: \['negativeClassLabel'\]"):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 3:
        with pytest.raises(
            DrumCommonException,
            match="Error - for multiclass classification, either the class labels or a class labels file must be provided in model-metadata.yaml file",
        ):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 4:
        with pytest.raises(
            DrumCommonException,
            match="Error - for multiclass classification, either the class labels or a class labels file should be provided in model-metadata.yaml file, but not both",
        ):
            read_model_metadata_yaml(tmp_path)
    elif test_case_number == 100:
        read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize(
    "hyper_param_metadata",
    [
        "complete_int_hyper_param",
        "complete_float_hyper_param",
        "complete_string_hyper_param",
        "complete_select_hyper_param",
        "complete_multi_hyper_param",
    ],
)
@pytest.mark.parametrize(
    "param_name_value",
    [
        "param",
        "a" * (PARAM_NAME_MAX_LENGTH),
        "a" * (PARAM_NAME_MAX_LENGTH - 1),
        "param_param",
        "param__param",
    ],
)
def test_yaml_metadata__hyper_param_valid_name(
    request, param_name_value, hyper_param_metadata, tmp_path, basic_model_metadata_yaml,
):
    hyper_param_metadata = request.getfixturevalue(hyper_param_metadata)
    model_metadata = basic_model_metadata_yaml
    hyper_param_metadata[ModelMetadataKeys.NAME] = param_name_value
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [hyper_param_metadata]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]["name"] == param_name_value


@pytest.mark.parametrize(
    "hyper_param_metadata",
    [
        "complete_int_hyper_param",
        "complete_float_hyper_param",
        "complete_string_hyper_param",
        "complete_select_hyper_param",
        "complete_multi_hyper_param",
    ],
)
@pytest.mark.parametrize(
    "param_name_value, error",
    [
        ("_param", "The parameter name should start or end with the Eng character."),
        ("param_", "The parameter name should start or end with the Eng character."),
        (
            "1" * (PARAM_NAME_MAX_LENGTH + 1),
            "Invalid parameter name: String is longer than 64 characters",
        ),
        ("特徴", "Only Eng characters and underscore are allowed."),
    ],
)
def test_yaml_metadata__hyper_param_invalid_name(
    request, param_name_value, error, hyper_param_metadata, tmp_path, basic_model_metadata_yaml,
):
    hyper_param_metadata = request.getfixturevalue(hyper_param_metadata)
    model_metadata = basic_model_metadata_yaml
    hyper_param_metadata[ModelMetadataKeys.NAME] = param_name_value
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [hyper_param_metadata]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match=error):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("required_field", ["name", "type", "min", "max"])
def test_yaml_metadata__int_hyper_param_required_fields(
    required_field, complete_int_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_int_hyper_param.pop(required_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_int_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match='"{}": "is required"'.format(required_field)):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("optional_field", ["default"])
def test_yaml_metadata__int_hyper_param_optional_fields(
    optional_field, complete_int_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_int_hyper_param.pop(optional_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_int_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert optional_field not in model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]


@pytest.mark.parametrize(
    "hyper_param_metadata",
    [
        "complete_int_hyper_param",
        "complete_float_hyper_param",
        "complete_string_hyper_param",
        "complete_select_hyper_param",
        "complete_multi_hyper_param",
    ],
)
@pytest.mark.parametrize("invalid_param_type", ["invalid_type"])
def test_yaml_metadata__hyper_param_invalid_type(
    request, invalid_param_type, hyper_param_metadata, tmp_path, basic_model_metadata_yaml,
):
    hyper_param_metadata = request.getfixturevalue(hyper_param_metadata)
    model_metadata = basic_model_metadata_yaml
    hyper_param_metadata["type"] = invalid_param_type
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [hyper_param_metadata]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match="{'type': 'is invalid'}"):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize(
    "min_max, is_valid", [({"min": 1, "max": 0}, False), ({"min": 0, "max": 1}, True),]
)
def test_yaml_metadata__int_hyper_param_min_max(
    min_max, is_valid, complete_int_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_int_hyper_param.update(min_max)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_int_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        if is_valid:
            read_model_metadata_yaml(tmp_path)
        else:
            with pytest.raises(
                DrumCommonException,
                match="Invalid int parameter param_int: min must be greater than or equal to max",
            ):
                read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("min_max", [{"min": 0.1}, {"max": 1.1}, {"min": "0.1"}, {"max": "1.1"},])
def test_yaml_metadata__int_hyper_param_invalid_type(
    min_max, complete_int_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_int_hyper_param.update(min_max)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_int_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match="value can't be converted to int"):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize(
    "hyper_param_config", [{"min": 1, "max": 2, "default": 0}, {"min": 1, "max": 2, "default": 3},]
)
def test_yaml_metadata__int_hyper_param_invalid_default_value(
    hyper_param_config, complete_int_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_int_hyper_param.update(hyper_param_config)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_int_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(
            DrumCommonException,
            match="Invalid int parameter param_int: values must be between \[1, 2\]",
        ):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("required_field", ["name", "type", "min", "max"])
def test_yaml_metadata__float_hyper_param_required_fields(
    required_field, complete_float_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_float_hyper_param.pop(required_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_float_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match='"{}": "is required"'.format(required_field)):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("optional_field", ["default"])
def test_yaml_metadata__float_hyper_param_optional_fields(
    optional_field, complete_float_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_float_hyper_param.pop(optional_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_float_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert optional_field not in model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]


@pytest.mark.parametrize(
    "min_max, is_valid", [({"min": 1.0, "max": 0.0}, False), ({"min": 0.0, "max": 1.0}, True),]
)
def test_yaml_metadata__float_hyper_param_min_max(
    min_max, is_valid, complete_float_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_float_hyper_param.update(min_max)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_float_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        if is_valid:
            read_model_metadata_yaml(tmp_path)
        else:
            with pytest.raises(
                DrumCommonException,
                match="Invalid float parameter param_float: min must be greater than or equal to max",
            ):
                read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize(
    "hyper_param_config",
    [{"min": 1.0, "max": 2.0, "default": 0.0}, {"min": 1.0, "max": 2.0, "default": 3.0},],
)
def test_yaml_metadata__float_hyper_param_invalid_default_value(
    hyper_param_config, complete_float_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_float_hyper_param.update(hyper_param_config)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_float_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(
            DrumCommonException,
            match="Invalid float parameter param_float: values must be between \[1.0, 2.0\]",
        ):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("required_field", ["name", "type"])
def test_yaml_metadata__string_hyper_param_required_fields(
    required_field, complete_string_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_string_hyper_param.pop(required_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_string_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match='"{}": "is required"'.format(required_field)):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("optional_field", ["default"])
def test_yaml_metadata__stirng_hyper_param_optional_fields(
    optional_field, complete_string_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_string_hyper_param.pop(optional_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_string_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert optional_field not in model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]


def test_yaml_metadata__string_hyper_param_invalid_value_size(
    complete_string_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_string_hyper_param["default"] = "1" * (PARAM_STRING_MAX_LENGTH + 1)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_string_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match="String is longer than 1024 characters"):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("required_field", ["name", "type", "values"])
def test_yaml_metadata__multi_hyper_param_required_fields(
    required_field, complete_multi_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_multi_hyper_param.pop(required_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_multi_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match='"{}": "is required"'.format(required_field)):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("optional_field", ["default"])
def test_yaml_metadata__multi_hyper_param_optional_fields(
    optional_field, complete_multi_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_multi_hyper_param.pop(optional_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_multi_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert optional_field not in model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]


@pytest.mark.parametrize(
    "optional_component_param_keys",
    itertools.combinations(ModelMetadataMultiHyperParamTypes.all(), 1),
)
def test_yaml_metadata__multi_hyper_param_optional_component_params(
    optional_component_param_keys, complete_multi_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    for param_key in optional_component_param_keys:
        complete_multi_hyper_param["values"].pop(param_key)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_multi_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert ModelMetadataMultiHyperParamTypes.all() - set(optional_component_param_keys) == set(
            model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]["values"].keys()
        )


@pytest.mark.parametrize("required_field", ["name", "type", "values"])
def test_yaml_metadata__select_hyper_param_required_fields(
    required_field, complete_select_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_select_hyper_param.pop(required_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_select_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match='"{}": "is required"'.format(required_field)):
            read_model_metadata_yaml(tmp_path)


@pytest.mark.parametrize("optional_field", ["default"])
def test_yaml_metadata__select_hyper_param_optional_fields(
    optional_field, complete_select_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_select_hyper_param.pop(optional_field)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_select_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert len(model_metadata[ModelMetadataKeys.HYPERPARAMETERS]) == 1
        assert optional_field not in model_metadata[ModelMetadataKeys.HYPERPARAMETERS][0]


def test_yaml_metadata__select_hyper_param_invalid_value_size(
    complete_select_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_select_hyper_param["values"] = ["1" * (PARAM_SELECT_VALUE_MAX_LENGTH + 1)]
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_select_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match="String is longer than 32 characters"):
            read_model_metadata_yaml(tmp_path)


def test_yaml_metadata__select_hyper_param_invalid_value_num(
    complete_select_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    model_metadata = basic_model_metadata_yaml
    complete_select_hyper_param["values"] = ["1"] * (PARAM_SELECT_NUM_VALUES_MAX_LENGTH + 1)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_select_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        with pytest.raises(DrumCommonException, match="list length is greater than 24"):
            read_model_metadata_yaml(tmp_path)


def test_read_model_metadata_properly_casts_typeschema(tmp_path, training_metadata_yaml):
    config_yaml = training_metadata_yaml + dedent(
        """
        typeSchema:
           input_requirements:
           - field: number_of_columns
             condition: IN
             value:
               - 1
               - 2
           - field: data_types
             condition: EQUALS
             value:
               - NUM
               - TXT
           output_requirements:
           - field: number_of_columns
             condition: IN
             value: 2
           - field: data_types
             condition: EQUALS
             value: NUM
        """
    )
    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)

    yaml_conf = read_model_metadata_yaml(tmp_path)
    output_reqs = yaml_conf["typeSchema"]["output_requirements"]
    input_reqs = yaml_conf["typeSchema"]["input_requirements"]

    value_key = "value"
    expected_as_int_list = next(
        (el for el in input_reqs if el["field"] == "number_of_columns")
    ).get(value_key)
    expected_as_str_list = next((el for el in input_reqs if el["field"] == "data_types")).get(
        value_key
    )
    expected_as_int = next((el for el in output_reqs if el["field"] == "number_of_columns")).get(
        value_key
    )
    expected_as_str = next((el for el in output_reqs if el["field"] == "data_types")).get(value_key)

    assert all(isinstance(el, int) for el in expected_as_int_list)
    assert all(isinstance(el, str) for el in expected_as_str_list)
    assert isinstance(expected_as_str_list, list)

    assert isinstance(expected_as_int, int)
    assert isinstance(expected_as_str, str)


def version_mocks():
    responses.add(
        responses.GET,
        "http://yess/version/",
        json={"major": 2, "versionString": "2.21", "minor": 21},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://yess/customModels/{}/versions/".format(modelID),
        json=version_response,
        status=200,
    )


def task_version_mock():
    responses.add(
        responses.POST,
        "http://yess/customTasks/{}/versions/".format(modelID),
        json=tasks_version_response,
        status=200,
    )


def mock_get_model(model_type="training", target_type="Regression"):
    body = {
        "customModelType": model_type,
        "id": modelID,
        "name": "1",
        "description": "1",
        "targetType": target_type,
        "deployments_count": "1",
        "created_by": "1",
        "updated": "1",
        "created": "1",
        "latestVersion": version_response,
    }
    if model_type == "inference":
        body["language"] = "Python"
        body["trainingDataAssignmentInProgress"] = False
    responses.add(
        responses.GET, "http://yess/customModels/{}/".format(modelID), json=body,
    )
    responses.add(
        responses.POST, "http://yess/customModels/".format(modelID), json=body,
    )


def mock_post_blueprint():
    responses.add(
        responses.POST,
        "http://yess/customTrainingBlueprints/",
        json={
            "userBlueprintId": "2",
            "custom_model": {"id": "1", "name": "1"},
            "custom_model_version": {"id": "1", "label": "1"},
            "execution_environment": {"id": "1", "name": "1"},
            "execution_environment_version": {"id": "1", "label": "1"},
            "training_history": [],
        },
    )
    responses.add(
        responses.POST,
        "http://yess/customTasks/",
        json={
            "id": modelID,
            "target_type": "Regression",
            "created": "1",
            "updated": "1",
            "name": "1",
            "description": "1",
            "language": "Python",
            "created_by": "1",
        },
    )


def mock_post_add_to_repository():
    responses.add(
        responses.POST,
        "http://yess/projects/{}/blueprints/fromUserBlueprint/".format(projectID),
        json={"id": "1"},
    )


def mock_get_env():
    responses.add(
        responses.GET,
        "http://yess/executionEnvironments/{}/".format(environmentID),
        json={
            "id": "1",
            "name": "hi",
            "latestVersion": {"id": "hii", "environment_id": environmentID, "build_status": "yes"},
        },
    )


def mock_train_model():
    responses.add(
        responses.POST,
        "http://yess/projects/{}/models/".format(projectID),
        json={},
        adding_headers={"Location": "the/moon"},
    )
    responses.add(
        responses.GET,
        "http://yess/projects/{}/modelJobs/the/".format(projectID),
        json={
            "is_blocked": False,
            "id": "55",
            "processes": [],
            "model_type": "fake",
            "project_id": projectID,
            "blueprint_id": "1",
        },
    )


@responses.activate
@pytest.mark.parametrize(
    "config_yaml",
    [
        "training_metadata_yaml",
        "training_metadata_yaml_with_proj",
        "inference_metadata_yaml",
        "inference_multiclass_metadata_yaml",
        "inference_multiclass_metadata_yaml_label_file",
    ],
)
@pytest.mark.parametrize("existing_model_id", [None, modelID])
def test_push(request, config_yaml, existing_model_id, multiclass_labels, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    if existing_model_id:
        config_yaml = config_yaml + "\nmodelID: {}".format(existing_model_id)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    version_mocks()
    task_version_mock()
    mock_post_blueprint()
    mock_post_add_to_repository()
    mock_get_model(model_type=config["type"], target_type=config["targetType"].capitalize())
    mock_get_env()
    mock_train_model()
    push_fn = _push_training if config["type"] == "training" else _push_inference
    push_fn(config, code_dir="", endpoint="http://Yess", token="okay")

    calls = responses.calls
    custom_tasks_or_models_path = "customTasks" if push_fn == _push_training else "customModels"
    if existing_model_id is None:
        assert (
            calls[1].request.path_url == "/{}/".format(custom_tasks_or_models_path)
            and calls[1].request.method == "POST"
        )
        if config["targetType"] == TargetType.MULTICLASS.value:
            sent_labels = json.loads(calls[1].request.body)["classLabels"]
            assert sent_labels == multiclass_labels
        call_shift = 1
    else:
        call_shift = 0
    assert (
        calls[call_shift + 1].request.path_url
        == "/{}/{}/versions/".format(custom_tasks_or_models_path, modelID)
        and calls[call_shift + 1].request.method == "POST"
    )
    if push_fn == _push_training:
        assert (
            calls[call_shift + 2].request.path_url == "/customTrainingBlueprints/"
            and calls[call_shift + 2].request.method == "POST"
        )
        if "trainingModel" in config:
            assert (
                calls[call_shift + 3].request.path_url
                == "/projects/{}/blueprints/fromUserBlueprint/".format(projectID)
                and calls[call_shift + 3].request.method == "POST"
            )
            assert (
                calls[call_shift + 4].request.path_url == "/projects/abc123/models/"
                and calls[call_shift + 4].request.method == "POST"
            )
            assert len(calls) == 6 + call_shift
        else:
            assert len(calls) == 3 + call_shift
    else:
        assert len(calls) == 2 + call_shift


@responses.activate
@pytest.mark.parametrize(
    "config_yaml", ["inference_binary_metadata_yaml_no_target_name",],
)
def test_push_no_target_name_in_yaml(request, config_yaml, tmp_path):
    config_yaml = request.getfixturevalue(config_yaml)
    config_yaml = config_yaml + "\nmodelID: {}".format(modelID)

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        f.write(config_yaml)
    config = read_model_metadata_yaml(tmp_path)

    options = Namespace(code_dir=tmp_path, model_config=config)
    with pytest.raises(DrumCommonException, match="Missing keys: \['targetName'\]"):
        drum_push(options)


def test_output_in_code_dir():
    code_dir = "/test/code/is/here"
    output_other = "/test/not/code"
    output_code_dir = "/test/code/is/here/output"
    assert not output_in_code_dir(code_dir, output_other)
    assert output_in_code_dir(code_dir, output_code_dir)


def test_output_dir_copy():
    with tempfile.TemporaryDirectory() as tempdir:
        # setup
        file = Path(tempdir, "test.py")
        file.touch()
        Path(tempdir, "__pycache__").mkdir()
        out_dir = Path(tempdir, "out")
        out_dir.mkdir()

        # test
        create_custom_inference_model_folder(tempdir, str(out_dir))
        assert Path(out_dir, "test.py").exists()
        assert not Path(out_dir, "__pycache__").exists()
        assert not Path(out_dir, "out").exists()


def test_read_structured_input_arrow_csv_na_consistency(tmp_path):
    """
    Test that N/A values (None, numpy.nan) are handled consistently when using
    CSV vs Arrow as a prediction payload format.
    1. Make CSV and Arrow prediction payloads from the same dataframe
    2. Read both payloads
    3. Assert the resulting dataframes are equal
    """

    # arrange
    df = pd.DataFrame({"col_int": [1, np.nan, None], "col_obj": ["a", np.nan, None]})

    csv_filename = os.path.join(tmp_path, "X.csv")
    with open(csv_filename, "w") as f:
        f.write(df.to_csv(index=False))

    arrow_filename = os.path.join(tmp_path, "X.arrow")
    with open(arrow_filename, "wb") as f:
        f.write(pyarrow.ipc.serialize_pandas(df).to_pybytes())

    # act
    csv_df = StructuredInputReadUtils.read_structured_input_file_as_df(csv_filename)
    arrow_df = StructuredInputReadUtils.read_structured_input_file_as_df(arrow_filename)

    # assert
    is_nan = lambda x: isinstance(x, float) and np.isnan(x)
    is_none = lambda x: x is None

    assert_frame_equal(csv_df, arrow_df)
    # `assert_frame_equal` doesn't make a difference between None and np.nan.
    # To do an exact comparison, compare None and np.nan "masks".
    assert_frame_equal(csv_df.applymap(is_nan), arrow_df.applymap(is_nan))
    assert_frame_equal(csv_df.applymap(is_none), arrow_df.applymap(is_none))


class TestJavaPredictor:
    # Verifying that correct code branch is taken depending on the data size.
    # As jp object is not properly configured, just check for the expected error message.
    @pytest.mark.parametrize(
        "data_size, error_message",
        [(2, "object has no attribute 'predict'"), (40000, "object has no attribute 'predictCSV'")],
    )
    def test_java_predictor_py4j_data(self, data_size, error_message):
        from datarobot_drum.drum.language_predictors.java_predictor.java_predictor import (
            JavaPredictor,
        )

        jp = JavaPredictor()
        with pytest.raises(AttributeError, match=error_message):
            jp._predict(binary_data=b"d" * data_size)

    @patch.object(JavaPredictor, "find_free_port")
    def test_run_java_server_entry_point_fail(self, mock_find_free_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            mock_find_free_port.return_value = s.getsockname()[1]

            pred = JavaPredictor()
            pred.model_artifact_extension = ".jar"

            # check that PredictorEntryPoint can not bind to port as it is taken
            with pytest.raises(DrumCommonException, match="java gateway failed to start"):
                pred._run_java_server_entry_point()

            # check that JavaGateway() fails to connect
            with pytest.raises(DrumCommonException, match="Failed to connect to java gateway"):
                pred._setup_py4j_client_connection()

    def test_run_java_server_entry_point_succeed(self):
        pred = JavaPredictor()
        pred.model_artifact_extension = ".jar"
        pred._run_java_server_entry_point()
        # required to properly shutdown py4j Gateway
        pred._setup_py4j_client_connection()
        pred._stop_py4j()


@pytest.mark.parametrize("data_dtype,label_dtype", [(int, int), (float, int), (int, float)])
def test_sklearn_predictor_wrong_dtype_labels(data_dtype, label_dtype):
    """
    This test makes sure that the target values can be ints, and the class labels be floats, and
    everything still works okay
    """
    X = pd.DataFrame({"col1": range(10)})
    y = pd.Series(data=[data_dtype(0)] * 5 + [data_dtype(1)] * 5)
    csv_bytes = bytes(X.to_csv(index=False), encoding="utf-8")
    estimator = LogisticRegression()
    estimator.fit(X, y)
    adapter = PythonModelAdapter(model_dir=None, target_type=TargetType.BINARY)
    adapter._predictor_to_use = SKLearnPredictor()
    preds, cols = adapter.predict(
        estimator,
        positive_class_label=str(label_dtype(0)),
        negative_class_label=str(label_dtype(1)),
        binary_data=csv_bytes,
        target_type=TargetType.BINARY,
    )
    marshalled_cols = _marshal_labels(
        request_labels=[str(label_dtype(1)), str(label_dtype(0))], model_labels=cols,
    )
    assert marshalled_cols == [str(label_dtype(0)), str(label_dtype(1))]


def test_endswith_extension_ignore_case():
    assert DrumUtils.endswith_extension_ignore_case("f.ExT", ".eXt")
    assert DrumUtils.endswith_extension_ignore_case("f.pY", [".Py", ".Java"])
    assert not DrumUtils.endswith_extension_ignore_case("f.py", [".R"])


def test_find_files_by_extension(tmp_path):
    exts = [".ext", ".Rds", ".py"]
    Path(f"{tmp_path}/file.ext").touch()
    Path(f"{tmp_path}/file.RDS").touch()
    Path(f"{tmp_path}/file.PY").touch()
    Path(f"{tmp_path}/file.pY").touch()
    assert 4 == len(DrumUtils.find_files_by_extensions(tmp_path, exts))


def test_filename_exists_and_is_file(tmp_path, caplog):
    Path(f"{tmp_path}/custom.py").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.py")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.r").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.R").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.r").touch()
    Path(f"{tmp_path}/custom.R").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.r", "custom.R")
    assert "Found filenames that case-insensitively match each other" in caplog.text
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    caplog.clear()

    Path(f"{tmp_path}/custom.jl").touch()
    assert DrumUtils.filename_exists_and_is_file(tmp_path, "custom.jl")
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    Path(f"{tmp_path}/custom.PY").touch()
    assert not DrumUtils.filename_exists_and_is_file(tmp_path, "custom.py")
    assert "Found filenames that case-insensitively match expected filenames" in caplog.text
    assert "Found: ['custom.PY']" in caplog.text
    assert "Expected one of: ['custom.py']" in caplog.text
    for f in glob.glob(f"{tmp_path}/*"):
        os.remove(f)

    caplog.clear()


def test_predictor_extension():
    assert SKLearnPredictor().is_artifact_supported("artifact.PKL")
    assert XGBoostPredictor().is_artifact_supported("artifact.PkL")
    assert PyTorchPredictor().is_artifact_supported("artifact.pTh")
    assert KerasPredictor().is_artifact_supported("artifact.h5")
    assert PMMLPredictor().is_artifact_supported("artifact.PmMl")
    assert not SKLearnPredictor().is_artifact_supported("artifact.jar")
    assert not XGBoostPredictor().is_artifact_supported("artifact.jar")
    assert not PyTorchPredictor().is_artifact_supported("artifact.Jar")
    assert not KerasPredictor().is_artifact_supported("artifact.jaR")
    assert not PMMLPredictor().is_artifact_supported("artifact.jAr")


@pytest.mark.parametrize(
    "target_type, predictor_cls",
    itertools.product([TargetType.TRANSFORM,], [PythonPredictor, RPredictor, JavaPredictor],),
)
def test_validate_model_metadata_output_requirements(target_type, predictor_cls):
    """The validation on the specs defined in the output_requirements of model metadata is only triggered when the
    custom task is a transform task. This test checks this functionality.

    Expected results:
        - If the custom task is a transform task, the validation on the specs defined in the output_requirements will
          be triggered. In this case, the exception is raised due to the violation of output_requirements.
        - If the custom task is not a transform task, the validation on the spec defined in the output_requirements will
          not be triggered.
    """

    proba_pred_output = pd.DataFrame({"class_0": [0.1, 0.2, 0.3], "class_1": [0.9, 0.8, 0.7]})
    num_pred_output = pd.DataFrame(np.arange(10))
    predictor = predictor_cls()
    predictor._target_type = target_type
    type_schema = {
        "input_requirements": [{"field": "data_types", "condition": "IN", "value": "NUM"}],
        "output_requirements": [{"field": "data_types", "condition": "IN", "value": "CAT"}],
    }
    predictor._schema_validator = SchemaValidator(type_schema=type_schema)
    df_to_validate = (
        proba_pred_output
        if target_type.value in TargetType.CLASSIFICATION.value
        else num_pred_output
    )

    if target_type == TargetType.TRANSFORM:
        with pytest.raises(DrumSchemaValidationException) as ex:
            predictor._schema_validator.validate_outputs(df_to_validate)
        assert str(ex.value) == (
            "schema validation failed for output:\n ['Datatypes incorrect. Data has types: NUM, which includes values "
            "that are not in CAT.']"
        )
    else:
        predictor._schema_validator.validate_outputs(df_to_validate)


@pytest.mark.parametrize("class_ordering", [lambda x: x, lambda x: list(reversed(x))])
class TestReplaceSanitizedClassNames:
    def test_replace_sanitized_class_names_same_binary(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_positive_class_label = "a"
        r_pred._r_negative_class_label = "b"
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["a", "b"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a", "b"])

    def test_replace_sanitized_class_names_unsanitary_binary(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_positive_class_label = "a+1"
        r_pred._r_negative_class_label = "b+1"
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["a.1", "b.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a+1", "b+1"])

    def test_replace_sanitized_class_names_float_binary(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_positive_class_label = "7.0"
        r_pred._r_negative_class_label = "7.1"
        predictions = pd.DataFrame(np.ones((3, 2)), columns=class_ordering(["X7", "X7.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["7.0", "7.1"])

    def test_replace_sanitized_class_names_same_multiclass(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_class_labels = ["a", "b", "c"]
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a", "b", "c"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a", "b", "c"])

    def test_replace_sanitized_class_names_unsanitary_multiclass(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_class_labels = ["a+1", "b-1", "c$1"]
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a.1", "b.1", "c.1"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["a+1", "b-1", "c$1"])

    def test_replace_sanitized_class_names_float_multiclass(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_class_labels = ["7.0", "7.1", "7.2"]
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["X7", "X7.1", "X7.2"]))
        result = r_pred._replace_sanitized_class_names(predictions)
        assert list(result.columns) == class_ordering(["7.0", "7.1", "7.2"])

    def test_replace_sanitized_class_names_ambiguous_multiclass(self, class_ordering):
        r_pred = RPredictor()
        r_pred._r_class_labels = ["a+1", "a-1", "a$1"]
        predictions = pd.DataFrame(np.ones((3, 3)), columns=class_ordering(["a.1", "a.1", "a.1"]))
        with pytest.raises(DrumCommonException, match="Class label names are ambiguous"):
            r_pred._replace_sanitized_class_names(predictions)


def test_binary_class_labels_from_env():
    with DrumRuntime() as runtime:
        runtime.options = Namespace(
            negative_class_label="env0",
            positive_class_label="env1",
            class_labels=None,
            code_dir="",
            disable_strict_validation=False,
            logging_level="warning",
            subparser_name=RunMode.FIT,
            target_type=TargetType.BINARY,
            verbose=False,
            content_type=None,
            input=None,
            target_csv=None,
            target=None,
            row_weights=None,
            row_weights_csv=None,
            output=None,
            num_rows=0,
            sparse_column_file=None,
            parameter_file=None,
        )
        cmrunner = CMRunner(runtime)
        pipeline_str = cmrunner._prepare_fit_pipeline(run_language=RunLanguage.PYTHON)
        assert '"positiveClassLabel": "env1",' in pipeline_str
        assert '"negativeClassLabel": "env0",' in pipeline_str
        assert '"classLabels": null,' in pipeline_str


def test_binary_class_labels_from_target():
    test_data_path = os.path.join(TESTS_DATA_PATH, "iris_binary_training.csv")
    with DrumRuntime() as runtime:
        runtime.options = Namespace(
            negative_class_label=None,
            positive_class_label=None,
            class_labels=None,
            code_dir="",
            disable_strict_validation=False,
            logging_level="warning",
            subparser_name=RunMode.FIT,
            target_type=TargetType.BINARY,
            verbose=False,
            content_type=None,
            input=test_data_path,
            target_csv=None,
            target="Species",
            row_weights=None,
            row_weights_csv=None,
            output=None,
            num_rows=0,
            sparse_column_file=None,
            parameter_file=None,
        )
        cmrunner = CMRunner(runtime)
        pipeline_str = cmrunner._prepare_fit_pipeline(run_language=RunLanguage.PYTHON)
        assert '"positiveClassLabel": "Iris-setosa",' in pipeline_str
        assert '"negativeClassLabel": "Iris-versicolor",' in pipeline_str
        assert '"classLabels": null,' in pipeline_str
