from abc import ABC, abstractmethod
import base64
import logging
from enum import auto, Enum
from enum import Enum as PythonNativeEnum
from io import BytesIO
import operator
from typing import List, Type, TypeVar, Union

from PIL import Image
from strictyaml import Map, Optional, Seq, Int, Enum, Str, YAML
import numpy as np
import pandas as pd

from datarobot_drum.drum.exceptions import DrumSchemaValidationException

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
    """All acceptable values for the 'value' field. """

    NUM = auto()
    TXT = auto()
    CAT = auto()
    IMG = auto()
    DATE = auto()

    FORBIDDEN = auto()
    SUPPORTED = auto()
    REQUIRED = auto()

    NEVER = auto()
    DYNAMIC = auto()
    ALWAYS = auto()
    IDENTITY = auto()

    @classmethod
    def data_values(cls) -> List["Values"]:
        return [cls.NUM, cls.TXT, cls.IMG, cls.DATE, cls.CAT]

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
        super(DataTypes, self).__init__(condition, values)

    @staticmethod
    def is_text(x):
        """
        Decide if a pandas series is text, using a very simple heuristic:
        1. Count the number of elements in the series that contain 1 or more whitespace character
        2. If >75% of the elements have whitespace, the Series is text

        Parameters
        ----------
        x: pd.Series - Series to be analyzed for text

        Returns
        -------
        boolean: True for is text, False for not text
        """
        if pd.api.types.is_string_dtype(x) and pd.api.types.infer_dtype(x) != "boolean":
            pct_rows_with_whitespace = (x.str.count(r"\s") > 0).sum() / x.shape[0]
            return pct_rows_with_whitespace > 0.75
        return False

    @staticmethod
    def is_img(x):
        def convert(data):
            return Image.open(BytesIO(base64.b64decode(data)))

        try:
            x.apply(convert)
            return True
        except:
            return False

    @staticmethod
    def number_of_text_columns(X):
        return len(X.columns[list(X.apply(DataTypes.is_text, result_type="expand"))])

    @staticmethod
    def number_of_img_columns(X):
        return len(X.columns[list(X.apply(DataTypes.is_img, result_type="expand"))])

    def validate(self, dataframe):
        """A quirk of validation that follows the implementation of DataRobot is
        as follows: A condition `IN` requires that
        `set(types_present_in_dataframe) == set(self.values)`"""
        types = dict()
        types[Values.NUM] = dataframe.select_dtypes(np.number).shape[1] > 0
        txt_columns = self.number_of_text_columns(dataframe)
        img_columns = self.number_of_img_columns(dataframe)
        types[Values.TXT] = txt_columns > 0
        types[Values.IMG] = img_columns > 0
        types[Values.CAT] = (
            dataframe.select_dtypes("O").shape[1]
            - (txt_columns + img_columns)
            + dataframe.select_dtypes("boolean").shape[1]
            > 0
        )
        types[Values.DATE] = dataframe.select_dtypes("datetime").shape[1] > 0

        validation_errors = []

        types_present = [k for k, v in types.items() if v]

        base_error = f"Datatypes incorrect. Data has types: {types_present}"

        errors = {
            Conditions.EQUALS: f"{base_error}, but expected only {self.values[0]}.",
            Conditions.NOT_EQUALS: f"{base_error}, but expected {self.values[0]} to NOT be present.",
            Conditions.IN: f"{base_error}, but expected types to exactly match: {self.values}",
            Conditions.NOT_IN: f"{base_error}, but expected no types in: {self.values} to be present",
        }

        tests = {
            Conditions.EQUALS: lambda data_types: self.values == data_types,
            Conditions.NOT_EQUALS: lambda data_types: self.values[0] not in data_types,
            Conditions.IN: lambda data_types: set(self.values) == set(data_types),
            Conditions.NOT_IN: lambda data_types: all(el not in self.values for el in data_types),
        }

        if not tests[self.condition](types_present):
            return [errors[self.condition]]
        return []


class Sparsity(BaseValidator):
    def __init__(self, condition, values):
        super(Sparsity, self).__init__(condition, values)

    def validate(self, dataframe):

        is_sparse = dataframe.dtypes.apply(pd.api.types.is_sparse).any()

        sparse_input_allowed_values = [Values.SUPPORTED, Values.REQUIRED]
        sparse_output_allowed_values = [Values.DYNAMIC, Values.ALWAYS]

        dense_input_allowed_values = [Values.FORBIDDEN, Values.SUPPORTED]
        dense_output_allowed_values = [Values.NEVER, Values.DYNAMIC, Values.IDENTITY]

        value = self.values[0]

        if value in Values.input_values():
            io_type = "input"
        else:
            io_type = "output"

        if is_sparse and value not in sparse_output_allowed_values + sparse_input_allowed_values:
            return [
                f"Sparse {io_type} data found, however value is set to {value}, expecting dense"
            ]
        elif (
            not is_sparse and value not in dense_output_allowed_values + dense_input_allowed_values
        ):
            return [
                f"Dense {io_type} data found, however value is set to {value}, expecting sparse"
            ]
        else:
            return []


class NumColumns(BaseValidator):
    def __init__(self, condition, values):
        super(NumColumns, self).__init__(condition, values)

    def validate(self, dataframe):
        errors = []
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
                f"Num columns error, {n_columns} did not satisfy: {self.condition} {test_value}"
            ]
        return []


class ContainsMissing(BaseValidator):
    def __init__(self, condition, values):
        super(ContainsMissing, self).__init__(condition, values)

    def validate(self, dataframe):
        missing_output_disallowed = Values.NEVER
        missing_input_disallowed = Values.FORBIDDEN
        any_missing = dataframe.isna().any().any()

        value = self.values[0]

        if value in Values.input_values():
            io_type = "Input"
        else:
            io_type = "Output"

        if any_missing and value in [missing_output_disallowed, missing_input_disallowed]:
            return [f"{io_type} contains missing values, the model does not support missing."]
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
    are valid together while the initial validation only checks that the map is in the right general format."""

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

    def __init__(self, type_schema: dict, strict=True, verbose=False):
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
        return self._run_validate(dataframe, self._input_validators, "input")

    def validate_outputs(self, dataframe):
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
                logger.info("Schema validation completed for model {}.".format(step_label))
            return True
        else:
            logger.error("Schema validation found mismatch between dataset and the supplied schema")
            for error in errors:
                logger.error(error)
            if self.strict:
                raise DrumSchemaValidationException(
                    "schema validation failed for {}:\n {}".format(step_label, errors)
                )
            return False
