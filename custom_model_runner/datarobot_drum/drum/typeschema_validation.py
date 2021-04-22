from strictyaml import Map, Optional, Seq, Int, Enum, Str


class DataTypes(object):
    """Validation related to data types.  This is common between input and output."""

    FIELD = "data_types"
    VALUES = ["NUM", "TXT", "CAT", "IMG", "DATE"]
    CONDITIONS = ["EQUALS", "IN", "NOT_EQUALS", "NOT_IN"]
    _TYPES = Enum(VALUES)

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {
                "field": Enum(cls.FIELD),
                "condition": Enum(cls.CONDITIONS),
                "value": Seq(cls._TYPES) | cls._TYPES,
            }
        )

    def validate(self):
        raise NotImplementedError


class SparsityInput(object):
    FIELD = "sparse"
    VALUES = ["FORBIDDEN", "SUPPORTED", "REQUIRED", "UNKNOWN"]

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {"field": Enum(cls.FIELD), "condition": Enum("EQUALS"), "value": Enum(cls.VALUES),}
        )

    def validate(self):
        raise NotImplementedError


class SparsityOutput(object):
    FIELD = "sparse"
    VALUES = ["NEVER", "DYNAMIC", "ALWAYS", "UNKNOWN", "IDENTITY"]

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {"field": Enum(cls.FIELD), "condition": Enum("EQUALS"), "value": Enum(cls.VALUES),}
        )

    def validate(self):
        raise NotImplementedError


class NumColumns(object):
    FIELD = "number_of_columns"
    CONDITIONS = ["EQUALS", "IN", "NOT_EQUALS", "NOT_IN", "GREATER_THAN", "LESS_THAN"]

    @classmethod
    def get_yaml_validator(cls):
        return Map(
            {
                "field": Enum(cls.FIELD),
                "condition": Enum(cls.CONDITIONS),
                "value": Int() | Seq(Int()),
            }
        )

    def validate(self):
        raise NotImplementedError


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
            print(req.data)
            print(input_validation[req.data["field"]])
            req.revalidate(input_validation[req.data["field"]])
    if "output_requirements" in type_schema:
        for req in type_schema["output_requirements"]:
            req.revalidate(output_validation[req.data["field"]])
