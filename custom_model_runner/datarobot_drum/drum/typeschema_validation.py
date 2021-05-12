import base64
import logging
from enum import auto, Enum
from enum import Enum as PythonNativeEnum
from io import BytesIO
from typing import List

from PIL import Image
from strictyaml import Map, Optional, Seq, Int, Enum, Str, YAML
import numpy as np
import pandas as pd

from datarobot_drum.drum.exceptions import DrumSchemaValidationException

logger = logging.getLogger("drum." + __name__)


class Conditions:
    """All acceptable values for the 'condition' field."""

    EQUALS = "EQUALS"
    IN = "IN"
    NOT_EQUALS = "NOT_EQUALS"
    NOT_IN = "NOT_IN"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    NOT_GREATER_THAN = "NOT_GREATER_THAN"
    NOT_LESS_THAN = "NOT_LESS_THAN"


class Values:
    """All acceptable values for the 'value' field. """

    NUM = "NUM"
    TXT = "TXT"
    CAT = "CAT"
    IMG = "IMG"
    DATE = "DATE"
    FORBIDDEN = "FORBIDDEN"
    SUPPORTED = "SUPPORTED"
    REQUIRED = "REQUIRED"
    NEVER = "NEVER"
    DYNAMIC = "DYNAMIC"
    ALWAYS = "ALWAYS"
    IDENTITY = "IDENTITY"


class BaseValidator(object):
    FIELD = None
    CONDITIONS = None
    VALUES = None

    def __init__(self, condition, values):
        self.condition = condition
        self.values = values

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {
                "field": Enum(cls.FIELD),
                "condition": Enum(cls.CONDITIONS),
                "value": Enum(cls.VALUES),
            }
        )

    def validate(self, dataframe: pd.DataFrame):
        raise NotImplementedError


class DataTypes(BaseValidator):
    """Validation related to data types.  This is common between input and output."""

    FIELD = "data_types"
    VALUES = [Values.NUM, Values.TXT, Values.CAT, Values.IMG, Values.DATE]
    CONDITIONS = [Conditions.EQUALS, Conditions.IN, Conditions.NOT_EQUALS, Conditions.NOT_IN]
    _TYPES = Enum(VALUES)

    def __init__(self, condition, values):
        super().__init__(condition, values)
        if not isinstance(values, list):
            self.values = [values]
        if condition == "EQUALS" or condition == "NOT_EQUALS":
            if len(self.values) > 1:
                raise (Exception("Multiple values not supported, use EQUALS/NOT_EQUALS instead."))

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {
                "field": Enum(cls.FIELD),
                "condition": Enum(cls.CONDITIONS),
                "value": Seq(cls._TYPES) | cls._TYPES,
            }
        )

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
    def has_txt(X):
        return len(X.columns[list(X.apply(DataTypes.is_text, result_type="expand"))])

    @staticmethod
    def has_img(X):
        return len(X.columns[list(X.apply(DataTypes.is_img, result_type="expand"))])

    def validate(self, dataframe):
        types = dict()
        types["NUM"] = dataframe.select_dtypes(np.number).shape[1] > 0
        txt_len = self.has_txt(dataframe)
        img_len = self.has_img(dataframe)
        types["TXT"] = txt_len > 0
        types["IMG"] = img_len > 0
        types["CAT"] = (
            dataframe.select_dtypes("O").shape[1]
            - (txt_len + img_len)
            + dataframe.select_dtypes("boolean").shape[1]
            > 0
        )
        types["DATE"] = dataframe.select_dtypes("datetime").shape[1] > 0

        validation_errors = []

        if self.condition == Conditions.EQUALS:
            for dtype in self.VALUES:
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
        elif self.condition == Conditions.NOT_EQUALS:
            if types[self.values[0]]:
                validation_errors.append(
                    "Datatypes incorrect, {} was expected to not be present".format(self.values)
                )
        elif self.condition == Conditions.NOT_IN:
            for dtype in self.values:
                if types[dtype]:
                    validation_errors.append(
                        "Datatypes incorrect, {} was expected to not be present".format(self.values)
                    )

        elif self.condition == Conditions.IN:
            for dtype in self.VALUES:
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


class SparsityInput(BaseValidator):
    FIELD = "sparse"
    CONDITIONS = Conditions.EQUALS
    VALUES = [Values.FORBIDDEN, Values.SUPPORTED, Values.REQUIRED]

    def validate(self, dataframe):
        errors = []
        if dataframe.dtypes.apply(pd.api.types.is_sparse).any():
            if self.values not in [Values.SUPPORTED, Values.REQUIRED]:
                errors.append(
                    "Sparse input data found, however value is set to {}, expecting dense".format(
                        self.values
                    )
                )
        elif self.values not in [Values.FORBIDDEN, Values.SUPPORTED]:
            errors.append("Dense input data found, however value is set to {}, expecting sparse")
        return errors


class SparsityOutput(BaseValidator):
    FIELD = "sparse"
    CONDITIONS = Conditions.EQUALS
    VALUES = [Values.NEVER, Values.DYNAMIC, Values.ALWAYS, Values.IDENTITY]

    def validate(self, dataframe):
        errors = []
        if dataframe.dtypes.apply(pd.api.types.is_sparse).any():
            if self.values not in [Values.DYNAMIC, Values.ALWAYS]:
                errors.append(
                    "Sparse output data found, however value is set to {}, expecting dense".format(
                        self.values
                    )
                )
        elif self.values not in [Values.NEVER, Values.DYNAMIC, Values.IDENTITY]:
            errors.append("Dense output data found, however value is set to {}, expecting sparse")
        return errors


class NumColumns(BaseValidator):
    FIELD = "number_of_columns"
    CONDITIONS = [
        Conditions.EQUALS,
        Conditions.IN,
        Conditions.NOT_EQUALS,
        Conditions.NOT_IN,
        Conditions.GREATER_THAN,
        Conditions.LESS_THAN,
        Conditions.NOT_GREATER_THAN,
        Conditions.NOT_LESS_THAN,
    ]

    def __init__(self, condition, values):
        super().__init__(condition, values)
        if not isinstance(values, list):
            self.values = [values]

    def __init__(self, condition, values):
        self.condition = condition
        if not isinstance(values, list):
            self.values = [values]
        else:
            self.values = values
        self.values = [int(value) for value in self.values]

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {
                "field": Enum(cls.FIELD),
                "condition": Enum(cls.CONDITIONS),
                "value": Int() | Seq(Int()),
            }
        )

    def validate(self, dataframe):
        errors = []
        n_columns = len(dataframe.columns)
        if self.condition == Conditions.EQUALS:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns != self.values[0]:
                errors.append(
                    "Num columns error, found {} but expected {} columns".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == Conditions.IN:
            if not any([n_columns == value for value in self.values]):
                errors.append(
                    "Num columns error, found {} but expected number of columns to be in {}".format(
                        n_columns, self.values
                    )
                )
        elif self.condition == Conditions.NOT_EQUALS:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns == self.values[0]:
                errors.append(
                    "Num columns error, found {} columns, which is not supported".format(n_columns)
                )
        elif self.condition == Conditions.NOT_IN:
            if any([n_columns == value for value in self.values]):
                errors.append(
                    "Num columns error, found {} columns, which is not supported".format(n_columns)
                )
        elif self.condition == Conditions.GREATER_THAN:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns <= self.values[0]:
                errors.append(
                    "Num columns error, found {} columns, but expected greater than {}".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == Conditions.NOT_GREATER_THAN:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns > self.values[0]:
                errors.append(
                    "Num columns error, found {} columns, but expected greater than {}".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == Conditions.LESS_THAN:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns >= self.values[0]:
                errors.append(
                    "Num columns error, found {} columns but expected less than {}".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == Conditions.NOT_LESS_THAN:
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns < self.values[0]:
                errors.append(
                    "Num columns error, found {} columns but expected less than {}".format(
                        n_columns, self.values[0]
                    )
                )
        else:
            errors.append("Num columns error, unrecognized condition {}".format(self.condition))
        return errors


class InputContainsMissing(BaseValidator):
    FIELD = "contains_missing"
    CONDITIONS = Conditions.EQUALS
    VALUES = [Values.FORBIDDEN, Values.SUPPORTED]

    def validate(self, dataframe):
        any_missing = dataframe.isna().any().any()
        if any_missing and self.values == Values.FORBIDDEN:
            return ["Input contains missing values, the model does not support missing."]
        return []


class OutputContainsMissing(BaseValidator):
    FIELD = "contains_missing"
    CONDITIONS = Conditions.EQUALS
    VALUES = [Values.NEVER, Values.DYNAMIC]

    def validate(self, dataframe):
        any_missing = dataframe.isna().any().any()
        if any_missing and self.values == Values.NEVER:
            return ["Input contains missing values, the model does not support missing."]
        return []


def get_type_schema_yaml_validator() -> Map:
    seq_validator = Seq(
        Map(
            {
                "field": Enum(["data_types", "sparse", "number_of_columns", "contains_missing"]),
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
    input_validation = {
        d.FIELD: d.get_yaml_validator()
        for d in [DataTypes, SparsityInput, NumColumns, InputContainsMissing]
    }
    output_validation = {
        d.FIELD: d.get_yaml_validator()
        for d in [DataTypes, SparsityOutput, NumColumns, OutputContainsMissing]
    }
    if "input_requirements" in type_schema:
        for req in type_schema["input_requirements"]:
            field = EricFields.from_string(req.data["field"])
            req.revalidate(field.to_input_requirments())
            # req.revalidate(input_validation[req.data["field"]])
    if "output_requirements" in type_schema:
        for req in type_schema["output_requirements"]:
            print(req.data)
            field = EricFields.from_string(req.data["field"])
            req.revalidate(field.to_output_requiremenst())


class SchemaValidator:
    """
    SchemaValidator transforms the typeschema definition into usable validation objects to be used to verify the data
    meets the schema requirements.  Two methods, validate_inputs and validate_outputs are provided to perform the
    actual validation on the respective dataframes.
    """

    _input_validator_mapping = {
        DataTypes.FIELD: DataTypes,
        SparsityInput.FIELD: SparsityInput,
        NumColumns.FIELD: NumColumns,
        InputContainsMissing.FIELD: InputContainsMissing,
    }
    _output_validator_mapping = {
        DataTypes.FIELD: DataTypes,
        SparsityOutput.FIELD: SparsityOutput,
        NumColumns.FIELD: NumColumns,
        OutputContainsMissing.FIELD: OutputContainsMissing,
    }

    def __init__(self, type_schema: dict, strict=True, verbose=False):
        self._input_validators = [
            self._get_validator(schema, self._input_validator_mapping)
            for schema in type_schema.get("input_requirements", [])
        ]
        self._output_validators = [
            self._get_validator(schema, self._output_validator_mapping)
            for schema in type_schema.get("output_requirements", [])
        ]
        self.strict = strict
        self._verbose = verbose

    def _get_validator(self, schema, mapping):
        return mapping[schema["field"]](schema["condition"], schema["value"])

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


class EricConditions(PythonNativeEnum):
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
    def non_numeric(cls) -> List['EricConditions']:
        return [
            cls.EQUALS,
            cls.NOT_EQUALS,
            cls.IN,
            cls.NOT_IN,
        ]

    def __str__(self) -> str:
        return self.name


class EricValues(PythonNativeEnum):
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
    def data_values(cls) -> List['EricValues']:
        return [
            cls.NUM,
            cls.TXT,
            cls.IMG,
            cls.DATE,
            cls.CAT
        ]

    @classmethod
    def input_values(cls) -> List['EricValues']:
        return [
            cls.FORBIDDEN,
            cls.SUPPORTED,
            cls.REQUIRED
        ]

    @classmethod
    def output_values(cls) -> List['EricValues']:
        return [
            cls.NEVER,
            cls.DYNAMIC,
            cls.ALWAYS,
            cls.IDENTITY
        ]

    def __str__(self) -> str:
        return self.name


class EricFields(PythonNativeEnum):
    DATA_TYPES = auto()
    SPARSE = auto()
    NUMBER_OF_COLUMNS = auto()
    CONTAINS_MISSING = auto()

    def __str__(self) -> str:
        return self.name.lower()


    @classmethod
    def from_string(cls, field: str) -> 'EricFields':
        for el in list(cls):
            if str(el) == field:
                return el
        raise ValueError(f"No field matches: {field!r}")

    def conditions(self) -> List[EricConditions]:
        conditions = {
            EricFields.SPARSE: [EricConditions.EQUALS],
            EricFields.DATA_TYPES: EricConditions.non_numeric(),
            EricFields.NUMBER_OF_COLUMNS: list(EricConditions),
            EricFields.CONTAINS_MISSING: [EricConditions.EQUALS]
        }
        return conditions[self]

    def input_values(self) -> List[EricValues]:
        values = {
            EricFields.DATA_TYPES: EricValues.data_values(),
            EricFields.SPARSE: EricValues.input_values(),
            EricFields.NUMBER_OF_COLUMNS: [],
            EricFields.CONTAINS_MISSING: [EricValues.FORBIDDEN, EricValues.SUPPORTED]
        }
        return values[self]

    def output_values(self) -> List[EricValues]:
        values = {
            EricFields.DATA_TYPES: EricValues.data_values(),
            EricFields.SPARSE: EricValues.output_values(),
            EricFields.NUMBER_OF_COLUMNS: [],
            EricFields.CONTAINS_MISSING:[EricValues.NEVER, EricValues.DYNAMIC]
        }
        return values[self]

    def to_input_requirments(self):
        return get_mapping(self, self.input_values())

    def to_output_requiremenst(self):
        return get_mapping(self, self.output_values())


def get_mapping(field: EricFields, values: List[EricValues]):
    base_value_enum = Enum((str(el) for el in values))
    if field == EricFields.DATA_TYPES:
        value_enum = base_value_enum | Seq(base_value_enum)
    elif field == EricFields.NUMBER_OF_COLUMNS:
        value_enum = Int() | Seq(Int())
    else:
        value_enum = base_value_enum

    conditions = Enum((str(el) for el in field.conditions()))
    return Map(
        {
            "field": Enum(str(field)),
            "condition": conditions,
            "value": value_enum
        }
    )
