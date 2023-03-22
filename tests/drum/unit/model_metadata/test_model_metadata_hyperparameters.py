import copy
import os

import pytest
import yaml

from datarobot_drum.drum.common import read_model_metadata_yaml
from datarobot_drum.drum.drum import get_default_parameter_values
from datarobot_drum.drum.enum import (
    ModelMetadataKeys,
    MODEL_CONFIG_FILENAME,
    ModelMetadataMultiHyperParamTypes,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.model_metadata import (
    PARAM_NAME_MAX_LENGTH,
    PARAM_STRING_MAX_LENGTH,
    PARAM_SELECT_VALUE_MAX_LENGTH,
    PARAM_SELECT_NUM_VALUES_MAX_LENGTH,
    IntHyperParameterTrafaret,
    FloatHyperParameterTrafaret,
    MultiHyperParameterTrafaret,
)


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


class TestHyperParameterTrafaretTransform:
    def test_to_int(self, complete_int_hyper_param):
        actual = complete_int_hyper_param.copy()
        for k, v in actual.items():
            actual[k] = str(v)
        assert complete_int_hyper_param == IntHyperParameterTrafaret.transform(actual)

    def test_to_float(self, complete_float_hyper_param):
        actual = complete_float_hyper_param.copy()
        for k, v in actual.items():
            actual[k] = str(v)
        assert complete_float_hyper_param == FloatHyperParameterTrafaret.transform(actual)

    def test_to_multi(self, complete_multi_hyper_param):
        actual = copy.deepcopy(complete_multi_hyper_param)

        for value_key in ["int", "float"]:
            value_dict = actual["values"][value_key]
            for k, v in value_dict.items():
                value_dict[k] = str(v)

        assert complete_multi_hyper_param == MultiHyperParameterTrafaret.transform(actual)


@pytest.mark.parametrize(
    "model_metadata, default_value",
    [
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "int", "min": 0, "max": 1, "default": 1},
                ],
            },
            1,
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "int", "min": 0, "max": 1},
                ],
            },
            0,
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
                ],
            },
            1.0,
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "float", "min": 0.0, "max": 1.0},
                ],
            },
            0.0,
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "string", "default": "string"},
                ],
            },
            "string",
        ),
        ({ModelMetadataKeys.HYPERPARAMETERS: [{"name": "param_name", "type": "string"}]}, "",),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {
                        "name": "param_name",
                        "type": "select",
                        "values": ["value 1", "value 2"],
                        "default": "value 2",
                    }
                ],
            },
            "value 2",
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "select", "values": ["value 1", "value 2"],}
                ],
            },
            "value 1",
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {
                        "name": "param_name",
                        "type": "multi",
                        "values": {
                            "int": {"min": 0, "max": 1},
                            "select": {"values": ["value 1", "value 2"]},
                        },
                        "default": "value 2",
                    }
                ],
            },
            "value 2",
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {
                        "name": "param_name",
                        "type": "multi",
                        "values": {
                            "int": {"min": 0, "max": 1},
                            "select": {"values": ["value 1", "value 2"]},
                        },
                    }
                ],
            },
            0,
        ),
        (
            {
                ModelMetadataKeys.HYPERPARAMETERS: [
                    {"name": "param_name", "type": "multi", "values": {},}
                ],
            },
            None,
        ),
    ],
)
def test_get_default_parameter_values(model_metadata, default_value):
    if default_value is not None:
        assert get_default_parameter_values(model_metadata) == {"param_name": default_value}
    else:
        assert not get_default_parameter_values(model_metadata)


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
        "a",
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
        ("_param", "The parameter name should not start or end with the underscore."),
        ("_param_param", "The parameter name should not start or end with the underscore."),
        ("param_", "The parameter name should not start or end with the underscore."),
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
                match="Invalid int parameter param_int: max must be greater than min",
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
                match="Invalid float parameter param_float: max must be greater than min",
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
    "optional_component_param_key", list(ModelMetadataMultiHyperParamTypes.all_list()),
)
def test_yaml_metadata__multi_hyper_param_optional_component_params(
    optional_component_param_key, complete_multi_hyper_param, tmp_path, basic_model_metadata_yaml,
):
    """This test verifies the component parameter under the multi-type hyper parameter is default.
    """
    model_metadata = basic_model_metadata_yaml
    complete_multi_hyper_param["values"].pop(optional_component_param_key)
    model_metadata[ModelMetadataKeys.HYPERPARAMETERS] = [complete_multi_hyper_param]

    with open(os.path.join(tmp_path, MODEL_CONFIG_FILENAME), mode="w") as f:
        yaml.dump(model_metadata, f)
        model_metadata = read_model_metadata_yaml(tmp_path)
        assert ModelMetadataMultiHyperParamTypes.all() - set([optional_component_param_key]) == set(
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
