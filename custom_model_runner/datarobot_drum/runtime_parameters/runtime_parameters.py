"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import trafaret as t

from .runtime_parameters_schema import RuntimeParameterPayloadTrafaret
from .exceptions import InvalidJsonException
from .exceptions import InvalidRuntimeParam


class RuntimeParameters:
    """
    A class that is used to read runtime-parameters that are delivered to the executed
    custom model. The runtime parameters are supposed to be defined by the user via the DataRobot
    web UI. Do not try to bypass this class by writing your own proprietary code, because the
    internal implementation may change over time.
    """

    @staticmethod
    def get(key):
        runtime_param_key = f"MLOPS_RUNTIME_PARAM_{key}"
        if runtime_param_key not in os.environ:
            raise ValueError(f"Runtime parameter '{key}' does not exist!")

        try:
            env_value = json.loads(os.environ[runtime_param_key])
        except json.decoder.JSONDecodeError:
            raise InvalidJsonException(
                f"Invalid runtime parameter json payload. payload={os.environ[runtime_param_key]}"
            )

        try:
            transformed_env_value = RuntimeParameterPayloadTrafaret.check(env_value)
        except t.DataError as ex:
            raise InvalidRuntimeParam(
                f"Invalid runtime parameter! env_value={env_value}, exception: {str(ex)}, "
            )

        return transformed_env_value["payload"]
