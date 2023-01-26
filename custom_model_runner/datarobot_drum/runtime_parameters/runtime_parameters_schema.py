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
            {t.Key("payload"): t.Dict({t.Key("credential_type"): t.String}).allow_extra("*")},
        )


RuntimeParameterPayloadTrafaret = (
    RuntimeParameterStringPayloadTrafaret | RuntimeParameterCredentialPayloadTrafaret
)
