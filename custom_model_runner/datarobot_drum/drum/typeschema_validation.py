import base64
from io import BytesIO

from PIL import Image
from strictyaml import Map, Optional, Seq, Int, Enum, Str
import numpy as np
import pandas as pd


class DataTypes(object):
    """Validation related to data types.  This is common between input and output."""

    FIELD = "data_types"
    VALUES = ["NUM", "TXT", "CAT", "IMG", "DATE"]
    CONDITIONS = ["EQUALS", "IN", "NOT_EQUALS", "NOT_IN"]
    _TYPES = Enum(VALUES)
    _DTYPE_MAPPING = {}

    def __init__(self, condition, values):
        self.condition = condition
        if not isinstance(values, list):
            self.values = [values]
        else:
            self.values = values
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

        if self.condition == "EQUALS":
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
        elif self.condition == "NOT_EQUALS":
            if types[self.values[0]]:
                validation_errors.append(
                    "Datatypes incorrect, {} was expected to not be present".format(self.values)
                )
        elif self.condition == "NOT_IN":
            for dtype in self.values:
                if types[dtype]:
                    validation_errors.append(
                        "Datatypes incorrect, {} was expected to not be present".format(self.values)
                    )

        elif self.condition == "IN":
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


class SparsityInput(object):
    FIELD = "sparse"
    VALUES = ["FORBIDDEN", "SUPPORTED", "REQUIRED", "UNKNOWN"]

    def __init__(self, condition, values):
        self.condition = condition
        self.values = values

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {"field": Enum(cls.FIELD), "condition": Enum("EQUALS"), "value": Enum(cls.VALUES)}
        )

    def validate(self, dataframe):
        errors = []
        if dataframe.dtypes.apply(pd.api.types.is_sparse).any():
            if self.values not in ["SUPPORTED", "REQUIRED", "UNKNOWN"]:
                errors.append(
                    "Sparse input data found, however value is set to {}, expecting dense".format(
                        self.values
                    )
                )
        elif self.values not in ["FORBIDDEN", "SUPPORTED", "UNKNOWN"]:
            errors.append("Dense input data found, however value is set to {}, expecting sparse")
        return errors


class SparsityOutput(object):
    FIELD = "sparse"
    VALUES = ["NEVER", "DYNAMIC", "ALWAYS", "UNKNOWN", "IDENTITY"]

    def __init__(self, condition, values):
        self.condition = condition
        self.values = values

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {"field": Enum(cls.FIELD), "condition": Enum("EQUALS"), "value": Enum(cls.VALUES)}
        )

    def validate(self, dataframe):
        errors = []
        if dataframe.dtypes.apply(pd.api.types.is_sparse).any():
            if self.values not in ["DYNAMIC", "ALWAYS", "UNKNOWN"]:
                errors.append(
                    "Sparse output data found, however value is set to {}, expecting dense".format(
                        self.values
                    )
                )
        elif self.values not in ["NEVER", "DYNAMIC", "UNKNOWN", "IDENTITY"]:
            errors.append("Dense output data found, however value is set to {}, expecting sparse")
        return errors


class NumColumns(object):
    FIELD = "number_of_columns"
    CONDITIONS = ["EQUALS", "IN", "NOT_EQUALS", "NOT_IN", "GREATER_THAN", "LESS_THAN"]

    def __init__(self, condition, values):
        self.condition = condition
        if not isinstance(values, list):
            self.values = [values]
        else:
            self.values = values

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
        if self.condition == "EQUALS":
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns != self.values[0]:
                errors.append(
                    "Num columns error, found {} but expected {} columns".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == "IN":
            if not any([n_columns == value for value in self.values]):
                errors.append(
                    "Num columns error, found {} but expected number of columns to be in {}".format(
                        n_columns, self.values
                    )
                )
        elif self.condition == "NOT_EQUALS":
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns == self.values[0]:
                errors.append(
                    "Num columns error, found {} columns, which is not supported".format(n_columns)
                )
        elif self.condition == "NOT_IN":
            if any([n_columns == value for value in self.values]):
                errors.append(
                    "Num columns error, found {} columns, which is not supported".format(n_columns)
                )
        elif self.condition == "GREATER_THAN":
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns <= self.values[0]:
                errors.append(
                    "Num columns error, found {} columns, but expected greater than {}".format(
                        n_columns, self.values[0]
                    )
                )
        elif self.condition == "LESS_THAN":
            if len(self.values) > 1:
                errors.append("Num columns error, only one value can be accepted for EQUALS")
            elif n_columns >= self.values[0]:
                errors.append(
                    "Num columns error, found {} columns but expected less than {}".format(
                        n_columns, self.values[0]
                    )
                )
        else:
            errors.append("Num columns error, unrecognized condition {}".format(self.condition))
        return errors


def get_type_schema_yaml_validator():
    seq_validator = Seq(
        Map(
            {
                "field": Enum(["data_types", "sparse", "number_of_columns"]),
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


def revalidate_typeschema(type_schema):
    """Revalidation of the elements in the lists for input and output requirements is required because there are
    multiple valid sets of parameters and how the strictyaml validation works.  This checks that the provided values
    are valid together while the initial validation only checks that the map is in the right general format."""
    input_validation = {
        d.FIELD: d.get_yaml_validator() for d in [DataTypes, SparsityInput, NumColumns]
    }
    output_validation = {
        d.FIELD: d.get_yaml_validator() for d in [DataTypes, SparsityOutput, NumColumns]
    }
    if "input_requirements" in type_schema:
        for req in type_schema["input_requirements"]:
            req.revalidate(input_validation[req.data["field"]])
    if "output_requirements" in type_schema:
        for req in type_schema["output_requirements"]:
            req.revalidate(output_validation[req.data["field"]])
