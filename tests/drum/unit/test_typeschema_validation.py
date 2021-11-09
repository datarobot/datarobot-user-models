"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
from textwrap import dedent
from typing import List, Union

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import yaml
from strictyaml import load, YAMLValidationError

from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.typeschema_validation import (
    NumColumns,
    Conditions,
    get_type_schema_yaml_validator,
    revalidate_typeschema,
    Values,
    Fields,
    SchemaValidator,
    RequirementTypes,
    DataTypes,
    Sparsity,
    ContainsMissing,
)

from tests.drum.utils import test_data


def get_data(dataset_name: str) -> pd.DataFrame:
    test_data_dir = test_data()
    return pd.read_csv(test_data_dir / dataset_name)


CATS_AND_DOGS = get_data("cats_dogs_small_training.csv")
TEN_K_DIABETES = get_data("10k_diabetes.csv")
IRIS_BINARY = get_data("iris_binary_training.csv")
LENDING_CLUB = get_data("lending_club_reduced.csv")
BOSTON_PUBLIC_RAISES = get_data("BostonPublicRaises_80.csv")
TELCO_CHURN = get_data("telecomms_churn.csv")
IRIS_WITH_BOOL = get_data("iris_with_bool.csv")

CONDITIONS_EXCEPT_EQUALS = [
    Conditions.NOT_EQUALS,
    Conditions.IN,
    Conditions.NOT_IN,
    Conditions.GREATER_THAN,
    Conditions.LESS_THAN,
    Conditions.NOT_GREATER_THAN,
    Conditions.NOT_LESS_THAN,
]

VALUES_EXCEPT_FORBIDDEN_SUPPORTED = [
    Values.NUM,
    Values.TXT,
    Values.CAT,
    Values.IMG,
    Values.DATE,
    Values.REQUIRED,
    Values.NEVER,
    Values.DYNAMIC,
    Values.ALWAYS,
    Values.IDENTITY,
]

VALUES_EXCEPT_NEVER_DYNAMIC = [
    Values.NUM,
    Values.TXT,
    Values.CAT,
    Values.IMG,
    Values.DATE,
    Values.FORBIDDEN,
    Values.SUPPORTED,
    Values.REQUIRED,
    Values.ALWAYS,
    Values.IDENTITY,
]


@pytest.fixture
def lending_club():
    return LENDING_CLUB.copy()


@pytest.fixture
def iris_binary():
    return IRIS_BINARY.copy()


@pytest.fixture
def ten_k_diabetes():
    return TEN_K_DIABETES.copy()


@pytest.fixture
def cats_and_dogs():
    return CATS_AND_DOGS.copy()


@pytest.fixture
def boston_raises():
    return BOSTON_PUBLIC_RAISES.copy()


@pytest.fixture
def telco_churn_txt():
    data = TELCO_CHURN.copy()
    return data[["Churn", "tariff_plan_conds"]]


@pytest.fixture
def iris_with_bool():
    return IRIS_WITH_BOOL.copy()


def get_yaml_dict(condition, field, values, top_requirements: RequirementTypes) -> dict:
    def _get_val(value):
        if isinstance(value, Values):
            return str(value)
        return value

    if len(values) == 1:
        new_vals = _get_val(values[0])
    else:
        new_vals = [_get_val(el) for el in values]
    yaml_dict = {
        str(top_requirements): [
            {"field": str(field), "condition": str(condition), "value": new_vals}
        ]
    }
    return yaml_dict


def input_requirements_yaml(
    field: Fields, condition: Conditions, values: List[Union[int, Values]]
) -> str:
    yaml_dict = get_yaml_dict(condition, field, values, RequirementTypes.INPUT_REQUIREMENTS)
    return yaml.dump(yaml_dict)


def output_requirements_yaml(
    field: Fields, condition: Conditions, values: List[Union[int, Values]]
) -> str:
    yaml_dict = get_yaml_dict(condition, field, values, RequirementTypes.OUTPUT_REQUIREMENTS)
    return yaml.dump(yaml_dict)


class TestSchemaValidator:
    @pytest.fixture
    def data(self, iris_binary):
        yield iris_binary

    @pytest.fixture
    def missing_data(self, data):
        df = data.copy(deep=True)
        for col in df.columns:
            df.loc[df.sample(frac=0.1).index, col] = np.nan
        yield df

    @pytest.fixture
    def sparse_df(self):
        yield pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(sparse.eye(10)))

    @pytest.fixture
    def dense_df(self):
        yield pd.DataFrame(np.zeros((10, 10)))

    @staticmethod
    def yaml_str_to_schema_dict(yaml_str: str) -> dict:
        """this emulates how we cast a yaml to a dict for validation in
        `datarobot_drum.drum.common.read_model_metadata_yaml` and these assumptions
        are tested in: `tests.drum.test_units.test_read_model_metadata_properly_casts_typeschema` """
        schema = load(yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(schema)
        return schema.data

    def test_default_typeschema(self):
        validator = SchemaValidator(type_schema={}, use_default_type_schema=True)
        assert validator._using_default_type_schema

        # Ensure input validators are correctly set
        assert len(validator._input_validators) == 3
        assert isinstance(validator._input_validators[0], DataTypes)
        assert validator._input_validators[0].condition == Conditions.IN
        assert set(validator._input_validators[0].values) == {
            Values.DATE,
            Values.CAT,
            Values.TXT,
            Values.NUM,
        }

        assert isinstance(validator._input_validators[1], Sparsity)
        assert validator._input_validators[1].condition == Conditions.EQUALS
        assert validator._input_validators[1].values == [Values.FORBIDDEN]

        assert isinstance(validator._input_validators[2], ContainsMissing)
        assert validator._input_validators[2].condition == Conditions.EQUALS
        assert validator._input_validators[2].values == [Values.FORBIDDEN]

        # Ensure output validators are correctly set
        assert len(validator._output_validators) == 1
        assert isinstance(validator._output_validators[0], DataTypes)
        assert validator._output_validators[0].condition == Conditions.EQUALS
        assert validator._output_validators[0].values == [Values.NUM]

    @pytest.mark.parametrize(
        "condition, value, passing_dataset, passing_target, failing_dataset, failing_target",
        [
            (
                Conditions.IN,
                [Values.CAT, Values.NUM, Values.TXT],
                "iris_binary",
                "SepalLengthCm",
                "cats_and_dogs",
                "class",
            ),
            (
                Conditions.EQUALS,
                [Values.NUM],
                "iris_binary",
                "Species",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.NOT_IN,
                [Values.TXT],
                "iris_binary",
                "Species",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.NOT_EQUALS,
                [Values.CAT],
                "iris_binary",
                "Species",
                "lending_club",
                "is_bad",
            ),
            (
                Conditions.EQUALS,
                [Values.IMG],
                "cats_and_dogs",
                "class",
                "ten_k_diabetes",
                "readmitted",
            ),
            (
                Conditions.IN,
                [Values.CAT, Values.TXT],
                "boston_raises",
                "RAISE",
                "iris_binary",
                "Species",
            ),
            (
                Conditions.IN,
                [Values.CAT, Values.TXT],
                "telco_churn_txt",
                "Churn",
                "iris_binary",
                "Species",
            ),
            (Conditions.IN, [Values.NUM], "iris_with_bool", "Species", "cats_and_dogs", "class",),
        ],
        ids=lambda x: str([str(el) for el in x]) if isinstance(x, list) else str(x),
    )
    def test_data_types(
        self,
        condition,
        value,
        passing_dataset,
        passing_target,
        failing_dataset,
        failing_target,
        request,
    ):
        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        good_data = request.getfixturevalue(passing_dataset)
        good_data.drop(passing_target, inplace=True, axis=1)
        assert validator.validate_inputs(good_data)

        bad_data = request.getfixturevalue(failing_dataset)
        bad_data.drop(failing_target, inplace=True, axis=1)
        with pytest.raises(DrumSchemaValidationException):
            validator.validate_inputs(bad_data)

    @pytest.mark.parametrize("condition", Conditions.non_numeric())
    @pytest.mark.parametrize(
        "value, expected_value_count",
        [
            ([Values.COUNT_DICT], 0),
            ([Values.GEO], 0),
            ([Values.GEO, Values.DATE_DURATION], 0),
            ([Values.COUNT_DICT, Values.GEO], 0),
            ([Values.COUNT_DICT, Values.GEO, Values.NUM], 1),
        ],
    )
    def test_data_types_no_validation(self, condition, value, expected_value_count):
        """Test the data types that do not have associated validation"""
        if len(value) > 1 and condition in Conditions.single_value_conditions():
            return

        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)
        # check that the values without validators are not added to the validator
        # pylint: disable=protected-access
        assert len(validator._input_validators[0].values) == expected_value_count

    def test_data_types_no_validation_skips_validation(self, cats_and_dogs):
        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, Conditions.IN, [Values.GEO])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        good_data = cats_and_dogs
        good_data.drop("class", inplace=True, axis=1)
        assert validator.validate_inputs(good_data)

    def test_data_types_in_allows_extra(self, iris_binary):
        """Additional values should be allowed with the IN condition

        - field: data_types
          condition: IN
          value:
            - NUM
            - TXT
        """
        condition = Conditions.IN
        value = Values.data_values()

        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        validator.validate_inputs(iris_binary)

    def test_bytes_not_string(self):
        """Tests that bytes are not counted as a string, and don't error trying to regex against a bytearray.  This is
        a special case that is encountered when testing the output of transforms that output images"""
        img = np.random.bytes(32)
        assert not DataTypes.is_text(img)

    @pytest.mark.parametrize(
        "single_value_condition",
        [
            Conditions.EQUALS,
            Conditions.NOT_EQUALS,
            Conditions.GREATER_THAN,
            Conditions.NOT_GREATER_THAN,
            Conditions.LESS_THAN,
            Conditions.NOT_LESS_THAN,
        ],
    )
    def test_instantiating_validator_raises_error_for_too_many_values(
        self, single_value_condition, iris_binary
    ):
        yaml_str = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, single_value_condition, [1, 2])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        with pytest.raises(DrumSchemaValidationException):
            SchemaValidator(schema_dict)

    @pytest.mark.parametrize(
        "condition, value, fail_expected",
        [
            (Conditions.EQUALS, [6], False),
            (Conditions.EQUALS, [3], True),
            (Conditions.IN, [2, 4, 6], False),
            (Conditions.IN, [1, 2, 3], True),
            (Conditions.LESS_THAN, [7], False),
            (Conditions.LESS_THAN, [3], True),
            (Conditions.GREATER_THAN, [4], False),
            (Conditions.GREATER_THAN, [10], True),
            (Conditions.NOT_EQUALS, [5], False),
            (Conditions.NOT_EQUALS, [6], True),
            (Conditions.NOT_IN, [1, 2, 3], False),
            (Conditions.NOT_IN, [2, 4, 6], True),
            (Conditions.NOT_GREATER_THAN, [6], False),
            (Conditions.NOT_GREATER_THAN, [2], True),
            (Conditions.NOT_LESS_THAN, [3], False),
            (Conditions.NOT_LESS_THAN, [100], True),
        ],
        ids=lambda x: str(x),
    )
    def test_num_columns(self, data, condition, value, fail_expected):
        yaml_str = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)
        if fail_expected:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_inputs(data)
        else:
            assert validator.validate_inputs(data)

    @pytest.mark.parametrize(
        "value, missing_ok", [(Values.FORBIDDEN, False), (Values.SUPPORTED, True)]
    )
    def test_missing_input(self, data, missing_data, value, missing_ok):
        yaml_str = input_requirements_yaml(Fields.CONTAINS_MISSING, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        assert validator.validate_inputs(data)
        if missing_ok:
            assert validator.validate_inputs(missing_data)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_inputs(missing_data)

    @pytest.mark.parametrize("value, missing_ok", [(Values.NEVER, False), (Values.DYNAMIC, True)])
    def test_missing_output(self, data, missing_data, value, missing_ok):
        yaml_str = output_requirements_yaml(Fields.CONTAINS_MISSING, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        assert validator.validate_outputs(data)
        if missing_ok:
            assert validator.validate_outputs(missing_data)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator.validate_outputs(missing_data)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [
            (Values.FORBIDDEN, False, True),
            (Values.SUPPORTED, True, True),
            (Values.REQUIRED, True, False),
        ],
    )
    def test_sparse_input(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = input_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_inputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_inputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [
            (Values.NEVER, False, True),
            (Values.DYNAMIC, True, True),
            (Values.ALWAYS, True, False),
            (Values.IDENTITY, False, True),
        ],
    )
    def test_sparse_output(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = output_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_outputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_outputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok",
        [(Values.FORBIDDEN, False, True), (Values.REQUIRED, True, False),],
    )
    def test_multiple_input_requirements(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = input_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        num_input = input_requirements_yaml(
            Fields.DATA_TYPES, Conditions.EQUALS, [Values.NUM]
        ).replace("input_requirements:\n", "")
        random_output = output_requirements_yaml(
            Fields.NUMBER_OF_COLUMNS, Conditions.EQUALS, [10000]
        )
        yaml_str += num_input
        yaml_str += random_output
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_inputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_inputs, dense_df, should_pass=dense_ok)

    @pytest.mark.parametrize(
        "value, sparse_ok, dense_ok", [(Values.NEVER, False, True), (Values.ALWAYS, True, False),],
    )
    def test_multiple_output_requirements(self, sparse_df, dense_df, value, sparse_ok, dense_ok):
        yaml_str = output_requirements_yaml(Fields.SPARSE, Conditions.EQUALS, [value])
        num_output = output_requirements_yaml(
            Fields.DATA_TYPES, Conditions.EQUALS, [Values.NUM]
        ).replace("output_requirements:\n", "")
        random_input = input_requirements_yaml(Fields.NUMBER_OF_COLUMNS, Conditions.EQUALS, [10000])
        yaml_str += num_output
        yaml_str += random_input
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        self._assert_validation(validator.validate_outputs, sparse_df, should_pass=sparse_ok)
        self._assert_validation(validator.validate_outputs, dense_df, should_pass=dense_ok)

    @staticmethod
    def _assert_validation(validator_method, data_frame, should_pass):
        if should_pass:
            assert validator_method(data_frame)
        else:
            with pytest.raises(DrumSchemaValidationException):
                validator_method(data_frame)

    def test_data_types_error_message(self, ten_k_diabetes):
        """This tests the error formatting for the list of Values"""
        condition = Conditions.IN
        values = [Values.CAT, Values.NUM]
        target = "readmitted"

        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, values)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)
        ten_k_diabetes.drop(target, inplace=True, axis=1)

        match_str = r"has types:( \w+,?){2,3}.*, which includes values that are not in CAT, NUM"
        with pytest.raises(DrumSchemaValidationException, match=match_str):
            validator.validate_inputs(ten_k_diabetes)


class TestRevalidateTypeSchemaDataTypes:
    field = Fields.DATA_TYPES

    @pytest.mark.parametrize("condition", Conditions.non_numeric())
    def test_datatypes_allowed_conditions(self, condition):
        values = [Values.NUM, Values.TXT]
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize(
        "condition",
        [
            Conditions.GREATER_THAN,
            Conditions.LESS_THAN,
            Conditions.NOT_GREATER_THAN,
            Conditions.NOT_LESS_THAN,
        ],
    )
    def test_datatypes_unallowed_conditions(self, condition):
        values = [Values.NUM, Values.TXT]
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.data_values())
    def test_datatyped_allowed_values(self, value):
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, [value])
        output_data_type_str = output_requirements_yaml(self.field, condition, [value])

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.input_values() + Values.output_values())
    def test_datatypes_unallowed_values(self, value):
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, [value])
        output_data_type_str = output_requirements_yaml(self.field, condition, [value])

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)

    def test_datatypes_multiple_values(self):
        condition = Conditions.IN
        values = Values.data_values()
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize(
        "permutation",
        [[Values.CAT, Values.NUM], [Values.NUM, Values.CAT]],
        ids=lambda x: str([str(el) for el in x]),
    )
    def test_regression_test_datatypes_multi_values(self, permutation):
        corner_case = input_requirements_yaml(Fields.DATA_TYPES, Conditions.IN, permutation)
        parsed_yaml = load(corner_case, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    def test_datatypes_mix_allowed_and_unallowed_values(self):
        values = [Values.NUM, Values.REQUIRED]
        condition = Conditions.EQUALS
        input_data_type_str = input_requirements_yaml(self.field, condition, values)
        output_data_type_str = output_requirements_yaml(self.field, condition, values)

        for data_type_str in (input_data_type_str, output_data_type_str):
            parsed_yaml = load(data_type_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaSparse:
    field = Fields.SPARSE

    @pytest.mark.parametrize("value", Values.input_values())
    def test_sparsity_input_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.data_values() + Values.output_values())
    def test_sparsity_input_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_sparsity_input_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, Values.input_values())

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.output_values())
    def test_sparsity_output_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", Values.data_values() + Values.input_values())
    def test_sparsity_output_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_sparsity_output_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, Values.output_values())

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("condition", CONDITIONS_EXCEPT_EQUALS)
    def test_sparsity_input_output_disallows_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [Values.REQUIRED])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [Values.ALWAYS])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaContainsMissing:
    field = Fields.CONTAINS_MISSING

    @pytest.mark.parametrize("value", [Values.FORBIDDEN, Values.SUPPORTED])
    def test_contains_missing_input_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", VALUES_EXCEPT_FORBIDDEN_SUPPORTED)
    def test_contains_missing_input_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_contains_missing_input_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = input_requirements_yaml(
            self.field, condition, [Values.FORBIDDEN, Values.SUPPORTED]
        )

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", [Values.NEVER, Values.DYNAMIC])
    def test_contains_missing_output_allowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", VALUES_EXCEPT_NEVER_DYNAMIC)
    def test_contains_missing_output_disallowed_values(self, value):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(self.field, condition, [value])

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_contains_missing_output_only_single_value(self):
        condition = Conditions.EQUALS
        sparse_yaml_str = output_requirements_yaml(
            self.field, condition, [Values.NEVER, Values.DYNAMIC]
        )

        parsed_yaml = load(sparse_yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("condition", CONDITIONS_EXCEPT_EQUALS)
    def test_contains_missing_input_output_disallows_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [Values.REQUIRED])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [Values.ALWAYS])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            with pytest.raises(YAMLValidationError):
                revalidate_typeschema(parsed_yaml)


class TestRevalidateTypeSchemaNumberOfColumns:
    field = Fields.NUMBER_OF_COLUMNS

    @pytest.mark.parametrize("condition", list(Conditions))
    def test_number_of_columns_can_use_all_conditions(self, condition):
        sparse_yaml_input_str = input_requirements_yaml(self.field, condition, [1])
        sparse_yaml_output_str = output_requirements_yaml(self.field, condition, [1])
        for yaml_str in (sparse_yaml_input_str, sparse_yaml_output_str):
            parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
            revalidate_typeschema(parsed_yaml)

    def test_number_of_columns_can_have_multiple_ints(self):
        yaml_str = input_requirements_yaml(self.field, Conditions.EQUALS, [1, 0, -1])
        parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("value", list(Values))
    def test_number_of_columns_cannot_use_other_values(self, value):
        yaml_str = input_requirements_yaml(self.field, Conditions.EQUALS, [value])
        parsed_yaml = load(yaml_str, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_revalidate_typescehma_mutates_yaml_num_columns_to_int(self):
        yaml_single_int = input_requirements_yaml(self.field, Conditions.EQUALS, [1])
        yaml_int_list = input_requirements_yaml(self.field, Conditions.EQUALS, [1, 2])
        parsed_single_int = load(yaml_single_int, get_type_schema_yaml_validator())
        parsed_int_list = load(yaml_int_list, get_type_schema_yaml_validator())

        def get_value(yaml):
            return yaml[str(RequirementTypes.INPUT_REQUIREMENTS)][0]["value"].data

        assert isinstance(get_value(parsed_single_int), str)
        assert isinstance(get_value(parsed_int_list)[0], str)

        revalidate_typeschema(parsed_single_int)
        revalidate_typeschema(parsed_int_list)

        assert isinstance(get_value(parsed_single_int), int)
        assert isinstance(get_value(parsed_int_list)[0], int)


class TestRevalidateTypeSchemaMixedCases:
    @pytest.fixture
    def passing_yaml_string(self):
        yield dedent(
            """
            input_requirements:
            - field: data_types
              condition: IN
              value:
                - NUM
            - field: sparse
              condition: EQUALS
              value: FORBIDDEN
            output_requirements:
            - field: data_types
              condition: EQUALS
              value: NUM
            - field: sparse
              condition: EQUALS
              value: NEVER
            """
        )

    def test_happy_path(self, passing_yaml_string):
        parsed_yaml = load(passing_yaml_string, get_type_schema_yaml_validator())
        revalidate_typeschema(parsed_yaml)

    @pytest.mark.parametrize("requirements_key", list(RequirementTypes))
    def test_failing_on_bad_requirements_key(self, requirements_key, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace(str(requirements_key), "oooooops")
        with pytest.raises(YAMLValidationError):
            load(bad_yaml, get_type_schema_yaml_validator())

    def test_failing_on_bad_field(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("sparse", "oooooops")
        with pytest.raises(YAMLValidationError):
            load(bad_yaml, get_type_schema_yaml_validator())

    def test_failing_on_bad_condition(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("EQUALS", "oooooops")
        parsed_yaml = load(bad_yaml, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)

    def test_failing_on_bad_value(self, passing_yaml_string):
        bad_yaml = passing_yaml_string.replace("NUM", "oooooops")
        parsed_yaml = load(bad_yaml, get_type_schema_yaml_validator())
        with pytest.raises(YAMLValidationError):
            revalidate_typeschema(parsed_yaml)


@pytest.mark.parametrize(
    "condition, value, fails",
    [
        (Conditions.IN, [1, 2], False),
        (Conditions.IN, [0, 3], True),
        (Conditions.NOT_IN, [0, 1, 2], False),
        (Conditions.NOT_IN, [-2, 3], True),
        (Conditions.NOT_LESS_THAN, [0], False),
        (Conditions.NOT_LESS_THAN, [-5], True),
        (Conditions.NOT_GREATER_THAN, [5], False),
        (Conditions.NOT_GREATER_THAN, [0], True),
        (Conditions.GREATER_THAN, [0], False),
        (Conditions.GREATER_THAN, [-10], True),
        (Conditions.LESS_THAN, [10], False),
        (Conditions.LESS_THAN, [0], True),
        (Conditions.EQUALS, [1], False),
        (Conditions.EQUALS, [0], True),
        (Conditions.NOT_EQUALS, [0], False),
        (Conditions.NOT_EQUALS, [-9], True),
    ],
)
def test_num_col_values(condition, value, fails):
    if fails:
        with pytest.raises(ValueError):
            NumColumns(condition, value)
    else:
        NumColumns(condition, value)
