"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import trafaret as t


from datarobot_drum.drum.enum import ModelMetadataHyperParamTypes


# Max length of a user-defined parameter
PARAM_NAME_MAX_LENGTH = 64

# Max length of a select value
PARAM_SELECT_VALUE_MAX_LENGTH = 32

# Max number of possible select values
PARAM_SELECT_NUM_VALUES_MAX_LENGTH = 24

# Max length of the value of a string/unicode parameter
PARAM_STRING_MAX_LENGTH = 1024


class ParamNameTrafaret(t.String):
    def __init__(self, *args, **kw_args):
        super(ParamNameTrafaret, self).__init__(*args, max_length=PARAM_NAME_MAX_LENGTH, **kw_args)

    def check_and_return(self, value):
        try:
            return super(ParamNameTrafaret, self).check_and_return(value)
        except t.DataError as e:
            error_msg = "Invalid parameter name: {}".format(str(e))
            raise t.DataError(error_msg)


IntHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("int"),
        t.Key("min"): t.ToInt,
        t.Key("max"): t.ToInt,
        t.Key("default", optional=True): t.ToInt,
    }
)

FloatHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("float"),
        t.Key("min"): t.ToFloat,
        t.Key("max"): t.ToFloat,
        t.Key("default", optional=True): t.ToFloat,
    }
)

StringHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("string"),
        t.Key("default", optional=True): t.String(
            max_length=PARAM_STRING_MAX_LENGTH, allow_blank=True
        ),
    }
)

SelectHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("select"),
        t.Key("values"): t.List(
            t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH, allow_blank=False),
            min_length=1,
            max_length=PARAM_SELECT_NUM_VALUES_MAX_LENGTH,
        ),
        t.Key("default", optional=True): t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH),
    }
)

# Multi only supports int, float, or select
MultiHyperParameterTrafaret = t.Dict(
    {
        t.Key("name"): ParamNameTrafaret(),
        t.Key("type"): t.Enum("multi"),
        t.Key("values"): t.Dict(
            {
                t.Key("int", optional=True): t.Dict(
                    {
                        t.Key("min"): t.ToInt,
                        t.Key("max"): t.ToInt,
                    }
                ),
                t.Key("float", optional=True): t.Dict(
                    {
                        t.Key("min"): t.ToFloat,
                        t.Key("max"): t.ToFloat,
                    }
                ),
                t.Key("select", optional=True): t.Dict(
                    {
                        t.Key("values"): t.List(
                            t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH, allow_blank=False),
                            min_length=1,
                            max_length=PARAM_SELECT_NUM_VALUES_MAX_LENGTH,
                        ),
                    }
                ),
            }
        ),
        t.Key("default", optional=True): t.Or(
            t.Int, t.Float, t.String(max_length=PARAM_SELECT_VALUE_MAX_LENGTH)
        ),
    }
)


HyperParameterTrafaret = {
    ModelMetadataHyperParamTypes.INT: IntHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.FLOAT: FloatHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.STRING: StringHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.SELECT: SelectHyperParameterTrafaret,
    ModelMetadataHyperParamTypes.MULTI: MultiHyperParameterTrafaret,
}
