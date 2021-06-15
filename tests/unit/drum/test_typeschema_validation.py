import os
from textwrap import dedent
from typing import List, Union

import pytest
import numpy as np
import pandas as pd
import scipy
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
)


def get_data(dataset_name: str) -> pd.DataFrame:
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
    return pd.read_csv(os.path.join(tests_data_path, dataset_name))


CATS_AND_DOGS = get_data("cats_dogs_small_training.csv")
TEN_K_DIABETES = get_data("10k_diabetes.csv")
IRIS_BINARY = get_data("iris_binary_training.csv")
LENDING_CLUB = get_data("lending_club_reduced.csv")


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
    tests_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))

    @pytest.fixture
    def data(self, iris_binary):
        yield iris_binary

    @pytest.fixture
    def missing_data(self, data):
        df = data.copy(deep=True)
        for col in df.columns:
            df.loc[df.sample(frac=0.1).index, col] = pd.np.nan
        yield df

    @pytest.fixture
    def sparse_df(self):
        yield pd.DataFrame.sparse.from_spmatrix(scipy.sparse.eye(10))

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

    @pytest.mark.parametrize(
        "condition, value, passing_dataset, passing_target, failing_dataset, failing_target",
        [
            (
                Conditions.IN,
                [Values.CAT, Values.NUM],
                "iris_binary",
                "SepalLengthCm",
                "ten_k_diabetes",
                "readmitted",
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
                "SepalLengthCm",
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

    def test_data_types_raises_error_if_all_type_in_in_are_not_present(self, iris_binary):
        """Because of how it's implemented in DataRobot,

        - field: data_types
          condition: IN
          value:
            - NUM
            - TXT

        requires that the DataFrame's set of types present _EQUALS_ the set: {NUM, TXT},
        but uses the condition: `IN`  :shrug:
        """
        condition = Conditions.IN
        value = Values.data_values()

        yaml_str = input_requirements_yaml(Fields.DATA_TYPES, condition, value)
        schema_dict = self.yaml_str_to_schema_dict(yaml_str)
        validator = SchemaValidator(schema_dict)

        with pytest.raises(DrumSchemaValidationException):
            validator.validate_inputs(iris_binary)

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

    @pytest.mark.parametrize("condition", list(set(Conditions) - set(Conditions.non_numeric())))
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

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.data_values())))
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

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.input_values())))
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

    @pytest.mark.parametrize("value", list(set(Values) - set(Values.output_values())))
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

    @pytest.mark.parametrize("condition", list(set(Conditions) - {Conditions.EQUALS}))
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

    @pytest.mark.parametrize("value", list(set(Values) - {Values.FORBIDDEN, Values.SUPPORTED}))
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

    @pytest.mark.parametrize("value", list(set(Values) - {Values.NEVER, Values.DYNAMIC}))
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

    @pytest.mark.parametrize("condition", list(set(Conditions) - {Conditions.EQUALS}))
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
