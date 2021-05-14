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


class EricConditions(BaseEnum):
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
    def non_numeric(cls) -> List["EricConditions"]:
        return [
            cls.EQUALS,
            cls.NOT_EQUALS,
            cls.IN,
            cls.NOT_IN,
        ]

    @classmethod
    def single_value_conditions(cls) -> List["EricConditions"]:
        return [
            cls.EQUALS,
            cls.NOT_EQUALS,
            cls.GREATER_THAN,
            cls.NOT_GREATER_THAN,
            cls.LESS_THAN,
            cls.NOT_LESS_THAN,
        ]


class EricValues(BaseEnum):
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
    def data_values(cls) -> List["EricValues"]:
        return [cls.NUM, cls.TXT, cls.IMG, cls.DATE, cls.CAT]

    @classmethod
    def input_values(cls) -> List["EricValues"]:
        return [cls.FORBIDDEN, cls.SUPPORTED, cls.REQUIRED]

    @classmethod
    def output_values(cls) -> List["EricValues"]:
        return [cls.NEVER, cls.DYNAMIC, cls.ALWAYS, cls.IDENTITY]


class EricFields(BaseEnum):
    DATA_TYPES = auto()
    SPARSE = auto()
    NUMBER_OF_COLUMNS = auto()
    CONTAINS_MISSING = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def conditions(self) -> List[EricConditions]:
        conditions = {
            EricFields.SPARSE: [EricConditions.EQUALS],
            EricFields.DATA_TYPES: EricConditions.non_numeric(),
            EricFields.NUMBER_OF_COLUMNS: list(EricConditions),
            EricFields.CONTAINS_MISSING: [EricConditions.EQUALS],
        }
        return conditions[self]

    def input_values(self) -> List[EricValues]:
        values = {
            EricFields.DATA_TYPES: EricValues.data_values(),
            EricFields.SPARSE: EricValues.input_values(),
            EricFields.NUMBER_OF_COLUMNS: [],
            EricFields.CONTAINS_MISSING: [EricValues.FORBIDDEN, EricValues.SUPPORTED],
        }
        return values[self]

    def output_values(self) -> List[EricValues]:
        values = {
            EricFields.DATA_TYPES: EricValues.data_values(),
            EricFields.SPARSE: EricValues.output_values(),
            EricFields.NUMBER_OF_COLUMNS: [],
            EricFields.CONTAINS_MISSING: [EricValues.NEVER, EricValues.DYNAMIC],
        }
        return values[self]

    def to_input_requirements(self):
        return get_mapping(self, self.input_values())

    def to_output_requirements(self):
        return get_mapping(self, self.output_values())

    def to_validator_class(self) -> Type["BaseValidator"]:
        classes = {
            EricFields.DATA_TYPES: DataTypes,
            EricFields.SPARSE: Sparsity,
            EricFields.NUMBER_OF_COLUMNS: NumColumns,
            EricFields.CONTAINS_MISSING: ContainsMissing,
        }
        return classes[self]


def get_mapping(field: EricFields, values: List[EricValues]):
    base_value_enum = Enum([str(el) for el in values])
    if field == EricFields.DATA_TYPES:
        value_enum = base_value_enum | Seq(base_value_enum)
    elif field == EricFields.NUMBER_OF_COLUMNS:
        value_enum = Int() | Seq(Int())
    else:
        value_enum = base_value_enum

    conditions = Enum([str(el) for el in field.conditions()])
    return Map({"field": Enum(str(field)), "condition": conditions, "value": value_enum})


class BaseValidator(ABC):
    def __init__(self, condition: EricConditions, values: List[Union[str, int]]):
        if len(values) > 1 and condition in EricConditions.single_value_conditions():
            raise DrumSchemaValidationException(
                f"{condition} only accepts a single value for: {values}"
            )

        def convert_value(value):
            if isinstance(value, int):
                return value
            return EricValues.from_string(value)

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
        types = dict()
        types[EricValues.NUM] = dataframe.select_dtypes(np.number).shape[1] > 0
        txt_columns = self.number_of_text_columns(dataframe)
        img_columns = self.number_of_img_columns(dataframe)
        types[EricValues.TXT] = txt_columns > 0
        types[EricValues.IMG] = img_columns > 0
        types[EricValues.CAT] = (
            dataframe.select_dtypes("O").shape[1]
            - (txt_columns + img_columns)
            + dataframe.select_dtypes("boolean").shape[1]
            > 0
        )
        types[EricValues.DATE] = dataframe.select_dtypes("datetime").shape[1] > 0

        validation_errors = []

        if self.condition == EricConditions.EQUALS:
            for dtype in types.keys():
                if dtype == self.values[0]:
                    if not types[dtype]:
                        validation_errors.append(
                            "Datatypes incorrect, expected data to have {}".format(self.values)
                        )
                else:
                    if types[dtype]:
                        validation_errors.append(
                            "Datatypes incorrect, unexpected type {} found, expected {}".format(
                                dtype, self.values
                            )
                        )
        elif self.condition == EricConditions.NOT_EQUALS:
            if types[self.values[0]]:
                validation_errors.append(
                    "Datatypes incorrect, {} was expected to not be present".format(self.values)
                )
        elif self.condition == EricConditions.NOT_IN:
            for dtype in self.values:
                if types[dtype]:
                    validation_errors.append(
                        "Datatypes incorrect, {} was expected to not be present".format(self.values)
                    )

        elif self.condition == EricConditions.IN:
            for dtype in types.keys():
                if dtype in self.values:
                    if not types[dtype]:
                        validation_errors.append(
                            "Datatypes incorrect, expected {} to be present".format(dtype)
                        )
                elif types[dtype]:
                    validation_errors.append(
                        "Datatypes incorrect, {} is not  expected to be present".format(dtype)
                    )
        return validation_errors


class Sparsity(BaseValidator):
    def __init__(self, condition, values):
        super(Sparsity, self).__init__(condition, values)

    def validate(self, dataframe):

        is_sparse = dataframe.dtypes.apply(pd.api.types.is_sparse).any()

        sparse_input_allowed_values = [EricValues.SUPPORTED, EricValues.REQUIRED]
        sparse_output_allowed_values = [EricValues.DYNAMIC, EricValues.ALWAYS]

        dense_input_allowed_values = [EricValues.FORBIDDEN, EricValues.SUPPORTED]
        dense_output_allowed_values = [EricValues.NEVER, EricValues.DYNAMIC, EricValues.IDENTITY]

        value = self.values[0]

        if value in EricValues.input_values():
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
            EricConditions.EQUALS: operator.eq,
            EricConditions.NOT_EQUALS: operator.ne,
            EricConditions.IN: lambda a, b: a in b,
            EricConditions.NOT_IN: lambda a, b: a not in b,
            EricConditions.GREATER_THAN: operator.gt,
            EricConditions.NOT_GREATER_THAN: operator.le,
            EricConditions.LESS_THAN: operator.lt,
            EricConditions.NOT_LESS_THAN: operator.ge,
        }

        test_value = self.values
        if self.condition in EricConditions.single_value_conditions():
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
        missing_output_disallowed = EricValues.NEVER
        missing_input_disallowed = EricValues.FORBIDDEN
        any_missing = dataframe.isna().any().any()

        value = self.values[0]

        if value in EricValues.input_values():
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
                "field": Enum([str(el) for el in EricFields]),
                "condition": Str(),
                "value": Str() | Seq(Str()),
            }
        )
    )
    return Map(
        {
            Optional("input_requirements"): seq_validator,
            Optional("output_requirements"): seq_validator,
        }
    )


def revalidate_typeschema(type_schema: YAML):
    """Perform validation on each dictionary in the both lists.  This is required due to limitations in strictyaml.  See
    the strictyaml documentation on revalidation for details.  This checks that the provided values
    are valid together while the initial validation only checks that the map is in the right general format."""

    for input_req in type_schema.get("input_requirements", []):
        field = EricFields.from_string(input_req.data["field"])
        requirements = field.to_input_requirements()
        input_req.revalidate(requirements)

    for output_req in type_schema.get("output_requirements", []):
        field = EricFields.from_string(output_req.data["field"])
        output_req.revalidate(field.to_output_requirements())


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
        field = EricFields.from_string(schema["field"])
        condition = EricConditions.from_string(schema["condition"])
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
