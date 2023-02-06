"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from enum import Enum

import trafaret as t


class RuntimeParameterTypes(Enum):
    STRING = "string"
    CREDENTIAL = "credential"


class NativeEnumTrafaret(t.Enum):
    def __init__(self, enum_type):
        self.variants = [t.value for t in enum_type]
        self.enum_type = enum_type

    def transform(self, value, context=None):
        self.check_value(value)
        return self.enum_type(value)


class RuntimeParameterPayloadBaseTrafaret(t.Dict):
    def __init__(self, param_type, definition):
        assert definition, "Valid trafaret definition must be provided!"
        base_definition = {t.Key("type"): t.Enum(param_type)}
        definition.update(base_definition)
        super().__init__(definition)


class RuntimeParameterStringPayloadTrafaret(RuntimeParameterPayloadBaseTrafaret):
    def __init__(self):
        super().__init__(RuntimeParameterTypes.STRING.value, {t.Key("payload"): t.String})


class RuntimeParameterCredentialPayloadTrafaret(RuntimeParameterPayloadBaseTrafaret):
    def __init__(self):
        super().__init__(
            RuntimeParameterTypes.CREDENTIAL.value,
            {t.Key("payload"): t.Dict({t.Key("credentialType"): t.String}).allow_extra("*")},
        )


RuntimeParameterPayloadTrafaret = (
    RuntimeParameterStringPayloadTrafaret | RuntimeParameterCredentialPayloadTrafaret
)


RuntimeParameterDefinitionTrafaret = t.Dict(
    {
        t.Key("fieldName", to_name="name"): t.String,
        t.Key("type"): NativeEnumTrafaret(RuntimeParameterTypes),
        t.Key("defaultValue", optional=True, default=None, to_name="default"): t.Any | t.Null,
    }
).ignore_extra("*")
