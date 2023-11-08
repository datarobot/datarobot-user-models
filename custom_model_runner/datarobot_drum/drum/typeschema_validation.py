"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numbers
import os
import sys
from abc import ABC, abstractmethod
import base64
import logging
from enum import auto
from enum import Enum as PythonNativeEnum
from io import BytesIO
import operator
from typing import List, Type, TypeVar, Union

from PIL import Image
from strictyaml import Map, Optional, Seq, Int, Enum, Str, YAML
import numpy as np
import pandas as pd

from datarobot_drum.drum.exceptions import DrumSchemaValidationException
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.utils.dataframe import is_sparse_dataframe

logger = logging.getLogger("drum." + __name__)


T = TypeVar("T")


class BaseEnum(PythonNativeEnum):
    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_string(cls: Type[T], enum_str: str) -> T:
        for el in list(cls):
            if str(el) == enum_str:
                return el
        raise ValueError(f"No enum value matches: {enum_str!r}")


class RequirementTypes(BaseEnum):
    INPUT_REQUIREMENTS = auto()
    OUTPUT_REQUIREMENTS = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Conditions(BaseEnum):
    """All acceptable values for the 'condition' field."""

    EQUALS = auto()
    IN = auto()
    NOT_EQUALS = auto()
    NOT_IN = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    NOT_GREATER_THAN = auto()
    NOT_LESS_THAN = auto()

    @classmethod
    def non_numeric(cls) -> List["Conditions"]:
        return [
            cls.EQUALS,
            cls.NOT_EQUALS,
            cls.IN,
            cls.NOT_IN,
        ]

    @classmethod
    def single_value_conditions(cls) -> List["Conditions"]:
        return [
            cls.EQUALS,
            cls.NOT_EQUALS,
            cls.GREATER_THAN,
            cls.NOT_GREATER_THAN,
            cls.LESS_THAN,
            cls.NOT_LESS_THAN,
        ]


class Values(BaseEnum):
    """All acceptable values for the 'value' field."""

    NUM = auto()
    TXT = auto()
    CAT = auto()
    IMG = auto()
    DATE = auto()
    DATE_DURATION = auto()
    COUNT_DICT = auto()
    GEO = auto()

    FORBIDDEN = auto()
    SUPPORTED = auto()
    REQUIRED = auto()

    NEVER = auto()
    DYNAMIC = auto()
    ALWAYS = auto()
    IDENTITY = auto()

    @classmethod
    def data_values(cls) -> List["Values"]:
        return [
            cls.NUM,
            cls.TXT,
            cls.IMG,
            cls.DATE,
            cls.CAT,
            cls.DATE_DURATION,
            cls.COUNT_DICT,
            cls.GEO,
        ]

    @classmethod
    def input_values(cls) -> List["Values"]:
        return [cls.FORBIDDEN, cls.SUPPORTED, cls.REQUIRED]

    @classmethod
    def output_values(cls) -> List["Values"]:
        return [cls.NEVER, cls.DYNAMIC, cls.ALWAYS, cls.IDENTITY]


class Fields(BaseEnum):
    DATA_TYPES = auto()
    SPARSE = auto()
    NUMBER_OF_COLUMNS = auto()
    CONTAINS_MISSING = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def conditions(self) -> List[Conditions]:
        conditions = {
            Fields.SPARSE: [Conditions.EQUALS],
            Fields.DATA_TYPES: Conditions.non_numeric(),
            Fields.NUMBER_OF_COLUMNS: list(Conditions),
            Fields.CONTAINS_MISSING: [Conditions.EQUALS],
        }
        return conditions[self]

    def input_values(self) -> List[Values]:
        values = {
            Fields.DATA_TYPES: Values.data_values(),
            Fields.SPARSE: Values.input_values(),
            Fields.NUMBER_OF_COLUMNS: [],
            Fields.CONTAINS_MISSING: [Values.FORBIDDEN, Values.SUPPORTED],
        }
        return values[self]

    def output_values(self) -> List[Values]:
        values = {
            Fields.DATA_TYPES: Values.data_values(),
            Fields.SPARSE: Values.output_values(),
            Fields.NUMBER_OF_COLUMNS: [],
            Fields.CONTAINS_MISSING: [Values.NEVER, Values.DYNAMIC],
        }
        return values[self]

    def to_requirements(self, requirement_type: RequirementTypes) -> Map:
        types = {
            RequirementTypes.INPUT_REQUIREMENTS: _get_mapping(self, self.input_values()),
            RequirementTypes.OUTPUT_REQUIREMENTS: _get_mapping(self, self.output_values()),
        }
        return types[requirement_type]

    def to_validator_class(self) -> Type["BaseValidator"]:
        classes = {
            Fields.DATA_TYPES: DataTypes,
            Fields.SPARSE: Sparsity,
            Fields.NUMBER_OF_COLUMNS: NumColumns,
            Fields.CONTAINS_MISSING: ContainsMissing,
        }
        return classes[self]


def _get_mapping(field: Fields, values: List[Values]) -> Map:
    base_value_enum = Enum([str(el) for el in values])
    if field == Fields.DATA_TYPES:
        value_enum = base_value_enum | Seq(base_value_enum)
    elif field == Fields.NUMBER_OF_COLUMNS:
        value_enum = Int() | Seq(Int())
    else:
        value_enum = base_value_enum

    conditions = Enum([str(el) for el in field.conditions()])
    return Map({"field": Enum(str(field)), "condition": conditions, "value": value_enum})


class BaseValidator(ABC):
    def __init__(self, condition: Conditions, values: List[Union[str, int]]):
        if len(values) > 1 and condition in Conditions.single_value_conditions():
            raise DrumSchemaValidationException(
                f"{condition} only accepts a single value for: {values}"
            )

        def convert_value(value):
            if isinstance(value, int):
                return value
            return Values.from_string(value)

        self.condition = condition
        self.values = [convert_value(value) for value in values]

    @abstractmethod
    def validate(self, dataframe: pd.DataFrame):
        raise NotImplementedError


class DataTypes(BaseValidator):
    """Validation related to data types.  This is common between input and output."""

    def __init__(self, condition, values):
        # We currently do not support DRUM validation for these values, but they are supported in DataRobot
        self._SKIP_VALIDATION = {
            Values.DATE_DURATION.name,
            Values.COUNT_DICT.name,
            Values.GEO.name,
        }
        values = list(set(values) - self._SKIP_VALIDATION)
        if len(values) == 0:
            logger.info(
                f"Values ({self.list_str(values)}) specified do not have runtime validation in DRUM, only within DataRobot."
            )
        super(DataTypes, self).__init__(condition, values)

    @staticmethod
    def list_str(l: list) -> str:
        """f-strings do not do a great job dealing with lists of objects.  The __str__ method isn't called on the
        contained objects, and the result is in [].  This provides the nicely formatted representation we want
        in the error message"""
        return ", ".join(sorted([str(x) for x in l]))

    @staticmethod
    def is_text(x: pd.Series) -> bool:
        """
        Decide if a pandas series is text, using a very simple heuristic:
        1. Count the number of elements in the series that contain 1 or more whitespace character
        2. If >75% of the elements have whitespace, and either there are more than 60 unique values or
            more than 5% of values are unique then the Series is considered to be text

        Parameters
        ----------
        x: pd.Series - Series to be analyzed for text

        Returns
        -------
        boolean: True for is text, False for not text
        """
        MIN_WHITESPACE_ROWS = 0.75  # percent
        MIN_UNIQUE_VALUES = 0.05  # percent
        if (
            pd.api.types.is_string_dtype(x)
            and pd.api.types.infer_dtype(x) != "boolean"
            and pd.api.types.infer_dtype(x) != "bytes"
        ):
            pct_rows_with_whitespace = (x.str.count(r"\s") > 0).sum() / x.shape[0]
            unique = x.nunique()
            pct_unique_values = unique / x.shape[0]
            return pct_rows_with_whitespace >= MIN_WHITESPACE_ROWS and (
                pct_unique_values >= MIN_UNIQUE_VALUES or unique >= 60
            )
        return False

    @staticmethod
    def is_img(x: pd.Series) -> bool:
        def is_number(value):
            return isinstance(value, numbers.Number)

        def convert(data):
            if is_number(data) and np.isnan(data):
                return np.nan
            return Image.open(BytesIO(base64.b64decode(data)))

        try:
            result = x.apply(convert)
            if np.all(result.apply(is_number)) and np.all(result.apply(np.isnan)):
                return False
            return True
        except Exception as e:
            return False

    @staticmethod
    def is_integer_numeric(x: pd.Series) -> bool:
        """Integer numerics can be considered categoricals.  They do not always get
        passed in as ints.  For example if there are NaN values in an integer column it will
        actually be handled as a float by pandas."""
        try:
            return np.all(x == x.astype(pd.Int64Dtype()))
        except:
            return False

    @staticmethod
    def number_of_text_columns(X: pd.DataFrame) -> int:
        return len(X.columns[list(X.apply(DataTypes.is_text, result_type="expand"))])

    @staticmethod
    def number_of_img_columns(X: pd.DataFrame) -> int:
        return len(X.columns[list(X.apply(DataTypes.is_img, result_type="expand"))])

    @staticmethod
    def number_of_integer_equivalent_numeric_columns(X: pd.DataFrame) -> int:
        return len(X.columns[list(X.apply(DataTypes.is_integer_numeric, result_type="expand"))])

    def validate(self, dataframe: pd.DataFrame) -> list:
        """Perform validation of the dataframe against the supplied specification."""
        if len(self.values) == 0:
            logger.info("Skipping type validation")
            return []
        types = dict()

        if is_sparse_dataframe(dataframe):
            # only numeric can be a csr or matrix market sparse matrix
            types[Values.NUM] = True
            types[Values.TXT] = False
            types[Values.IMG] = False
            types[Values.CAT] = False
            types[Values.DATE] = False
            num_possible_numeric_categorical = 0
            num_numeric = 1
        else:
            num_bool_columns = dataframe.select_dtypes("boolean").shape[1]
            num_txt_columns = self.number_of_text_columns(dataframe)
            num_img_columns = self.number_of_img_columns(dataframe)
            num_obj_columns = dataframe.select_dtypes("O").shape[1]
            # Note that boolean values will be sent as numeric in DataRobot
            if num_bool_columns > 0:
                logger.warning(
                    "Boolean values were present in the data, which are passed as numeric input in DataRobot.  You may need to convert boolean values to integers/floats for your model"
                )
            num_possible_numeric_categorical = self.number_of_integer_equivalent_numeric_columns(
                dataframe
            )
            num_numeric = dataframe.select_dtypes(np.number).shape[1]
            types[Values.NUM] = num_numeric > 0 or num_bool_columns > 0
            types[Values.TXT] = num_txt_columns > 0
            types[Values.IMG] = num_img_columns > 0
            types[Values.CAT] = num_obj_columns - num_img_columns
            types[Values.DATE] = dataframe.select_dtypes("datetime").shape[1] > 0

        types_present = [k for k, v in types.items() if v]

        base_error = f"Datatypes incorrect. Data has types: {DataTypes.list_str(types_present)}"

        errors = {
            Conditions.EQUALS: f"{base_error}, but expected types to exactly match: {DataTypes.list_str(self.values)}",
            Conditions.NOT_EQUALS: f"{base_error}, but expected {self.values[0]} to NOT be present.",
            Conditions.IN: f"{base_error}, which includes values that are not in {DataTypes.list_str(self.values)}.",
            Conditions.NOT_IN: f"{base_error}, but expected no types in: {DataTypes.list_str(self.values)} to be present",
        }

        # Treat Cat and TXT as equivalent
        remapped_values = [Values.CAT if k == Values.TXT else k for k in self.values]
        # Treat Cat and NUM as equivalent
        remapped_values = [
            Values.CAT if k == Values.NUM and num_possible_numeric_categorical == num_numeric else k
            for k in remapped_values
        ]
        tests = {
            Conditions.EQUALS: lambda data_types: set(remapped_values) == set(data_types),
            Conditions.NOT_EQUALS: lambda data_types: remapped_values[0] not in data_types,
            Conditions.IN: lambda data_types: set(data_types).issubset(set(remapped_values)),
            Conditions.NOT_IN: lambda data_types: all(
                el not in remapped_values for el in data_types
            ),
        }

        types_present = [Values.CAT if k == Values.TXT else k for k in types_present]
        types_present = [
            Values.CAT if k == Values.NUM and num_possible_numeric_categorical == num_numeric else k
            for k in types_present
        ]
        if not tests[self.condition](types_present):
            return [errors[self.condition]]
        return []


class Sparsity(BaseValidator):
    def __init__(self, condition, values):
        super(Sparsity, self).__init__(condition, values)

    def validate(self, dataframe):
        _is_sparse = is_sparse_dataframe(dataframe)

        sparse_input_allowed_values = [Values.SUPPORTED, Values.REQUIRED]
        sparse_output_allowed_values = [Values.DYNAMIC, Values.ALWAYS, Values.IDENTITY]

        dense_input_allowed_values = [Values.FORBIDDEN, Values.SUPPORTED]
        dense_output_allowed_values = [Values.NEVER, Values.DYNAMIC, Values.IDENTITY]

        value = self.values[0]

        if value in Values.input_values():
            io_type = "input"
        else:
            io_type = "output"

        if _is_sparse and value not in sparse_output_allowed_values + sparse_input_allowed_values:
            return [
                f"Sparse {io_type} data found, however value is set to {value}, expecting dense"
            ]
        elif (
            not _is_sparse and value not in dense_output_allowed_values + dense_input_allowed_values
        ):
            return [
                f"Dense {io_type} data found, however value is set to {value}, expecting sparse"
            ]
        else:
            return []


class NumColumns(BaseValidator):
    def __init__(self, condition, values):
        super(NumColumns, self).__init__(condition, values)
        if not all([v >= 0 for v in values]):
            raise ValueError("The value for number of columns can not be negative")
        if 0 in values:
            if condition not in [
                Conditions.NOT_IN,
                Conditions.NOT_EQUALS,
                Conditions.NOT_LESS_THAN,
                Conditions.GREATER_THAN,
            ]:
                raise ValueError(f"Value of 0 is not supported for {condition}")

    def validate(self, dataframe):
        n_columns = len(dataframe.columns)

        conditions_map = {
            Conditions.EQUALS: operator.eq,
            Conditions.NOT_EQUALS: operator.ne,
            Conditions.IN: lambda a, b: a in b,
            Conditions.NOT_IN: lambda a, b: a not in b,
            Conditions.GREATER_THAN: operator.gt,
            Conditions.NOT_GREATER_THAN: operator.le,
            Conditions.LESS_THAN: operator.lt,
            Conditions.NOT_LESS_THAN: operator.ge,
        }

        test_value = self.values
        if self.condition in Conditions.single_value_conditions():
            test_value = self.values[0]

        passes = conditions_map[self.condition](n_columns, test_value)
        if not passes:
            return [
                f"Incorrect number of columns. {n_columns} received. However, the schema dictates that number of columns should be {self.condition} {test_value}"
            ]

        return []


class ContainsMissing(BaseValidator):
    def __init__(self, condition, values):
        super(ContainsMissing, self).__init__(condition, values)

    def validate(self, dataframe):
        missing_output_disallowed = Values.NEVER
        missing_input_disallowed = Values.FORBIDDEN
        if is_sparse_dataframe(dataframe):
            # sparse but not NA...
            any_missing = False
        else:
            any_missing = dataframe.isna().any().any()

        value = self.values[0]

        if value in Values.input_values():
            io_type = "Input"
        else:
            io_type = "Output"

        if any_missing and value in [missing_output_disallowed, missing_input_disallowed]:
            return [
                f"{io_type} contains missing values, which the supplied task schema does not allow"
            ]
        return []


def get_type_schema_yaml_validator() -> Map:
    seq_validator = Seq(
        Map(
            {
                "field": Enum([str(el) for el in Fields]),
                "condition": Str(),
                "value": Str() | Seq(Str()),
            }
        )
    )
    return Map(
        {
            Optional(str(RequirementTypes.INPUT_REQUIREMENTS)): seq_validator,
            Optional(str(RequirementTypes.OUTPUT_REQUIREMENTS)): seq_validator,
        }
    )


def revalidate_typeschema(type_schema: YAML):
    """THIS MUTATES `type_schema`! calling the function would change {"number_of_columns": {"value": "1"}}
    to {"number_of_columns": {"value": 1}}

    Perform validation on each dictionary in the both lists.  This is required due to limitations in strictyaml.  See
    the strictyaml documentation on revalidation for details.  This checks that the provided values
    are valid together while the initial validation only checks that the map is in the right general format.
    """

    for requriment_type in RequirementTypes:
        for req in type_schema.get(str(requriment_type), []):
            field = Fields.from_string(req.data["field"])
            req.revalidate(field.to_requirements(requriment_type))


class SchemaValidator:
    """
    SchemaValidator transforms the typeschema definition into usable validation objects to be used to verify the data
    meets the schema requirements.  Two methods, validate_inputs and validate_outputs are provided to perform the
    actual validation on the respective dataframes.
    """

    _DEFAULT_TYPE_SCHEMA_CODEDIR_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "resource", "default_typeschema")
    )
    assert os.path.exists(_DEFAULT_TYPE_SCHEMA_CODEDIR_PATH)

    def __init__(
        self, type_schema: dict, strict=True, use_default_type_schema=False, verbose=False
    ):
        """
        Parameters
        ----------
        type_schema: dict
            YAML type schema converted to dict
        strict: bool
            Whether to error if data does not match type schema
        use_default_type_schema: bool
            Whether to use the default type schema which matches DataRobot's defaults when no type schema is present.
            type_schema must not be provided for the default to be used.
        verbose: bool
            Whether to print messages to the user
        """
        self._using_default_type_schema = False
        if not type_schema and use_default_type_schema:
            from datarobot_drum.drum.common import (
                read_model_metadata_yaml,
            )  # local import to prevent cyclic dependency

            type_schema = read_model_metadata_yaml(
                SchemaValidator._DEFAULT_TYPE_SCHEMA_CODEDIR_PATH
            )["typeSchema"]
            self._using_default_type_schema = True

        self._input_validators = [
            self._get_validator(schema) for schema in type_schema.get("input_requirements", [])
        ]
        self._output_validators = [
            self._get_validator(schema) for schema in type_schema.get("output_requirements", [])
        ]

        self.strict = strict
        self._verbose = verbose

    def _get_validator(self, schema):
        field = Fields.from_string(schema["field"])
        condition = Conditions.from_string(schema["condition"])
        values = schema["value"]
        if not isinstance(values, list):
            values = [values]
        return field.to_validator_class()(condition, values)

    def validate_inputs(self, dataframe):
        # Validate that the input values are of the type and shape the user specified in the schema
        return self._run_validate(dataframe, self._input_validators, "input")

    def validate_outputs(self, dataframe):
        # Validate that the output values are of the type and shape the user specified in the schema
        return self._run_validate(dataframe, self._output_validators, "output")

    def _run_validate(self, dataframe, validators, step_label):
        errors = []
        for validator in validators:
            errors.extend(validator.validate(dataframe))
        if len(validators) == 0:
            if self._verbose:
                logger.info("No type schema for {} provided.".format(step_label))
            return True
        elif len(errors) == 0:
            if self._verbose:
                logger.info("Schema validation completed for task {}.".format(step_label))
            return True
        else:
            logger.error(
                "Schema validation found mismatch between {} dataset and the supplied schema".format(
                    step_label
                )
            )
            for error in errors:
                logger.error(error)
            if self.strict:
                raise DrumSchemaValidationException(
                    "schema validation failed for {}:\n {}".format(step_label, errors)
                )
            return False

    def validate_type_schema(self, target_type):
        """Validate typeSchema section of model metadata.

        Parameters
        ----------
        target_type: TargetType
            Enum defined in TargetType.

        Raises
        ------
        DrumSchemaValidationException
            Raised when target type is not transform and output_requirements exists in the model metadata typeSchema.
        """
        if target_type != TargetType.TRANSFORM and self._output_validators:
            msg = "Specifying output_requirements in model_metadata.yaml is only valid for custom transform tasks."

            print(msg, file=sys.stderr)
            raise DrumSchemaValidationException(msg)
