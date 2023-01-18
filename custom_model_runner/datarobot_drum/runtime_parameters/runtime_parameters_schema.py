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


RuntimeParameterPayloadTrafaret = t.Dict(
    {
        t.Key("type"): t.Enum(RuntimeParameterTypes.STRING.value),
        t.Key("payload"): t.String(allow_blank=True),
    }
) | t.Dict(
    {
        t.Key("type"): t.Enum(RuntimeParameterTypes.CREDENTIAL.value),
        t.Key("payload"): t.Dict({t.Key("credential_type"): t.Enum("s3")}).allow_extra("*"),
    }
)
