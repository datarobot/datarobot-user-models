"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import re

import trafaret as t
import yaml

from .exceptions import InvalidJsonException
from .exceptions import InvalidRuntimeParam
from .runtime_parameters_schema import RuntimeParameterCredentialPayloadTrafaret
from .runtime_parameters_schema import RuntimeParameterPayloadTrafaret
from .runtime_parameters_schema import RuntimeParameterStringPayloadTrafaret
from .runtime_parameters_schema import RuntimeParameterTypes


class RuntimeParameters:
    """
    A class that is used to read runtime-parameters that are delivered to the executed
    custom model. The runtime parameters are supposed to be defined by the user via the DataRobot
    web UI. Do not try to bypass this class by writing your own proprietary code, because the
    internal implementation may change over time.
    """

    PARAM_PREFIX = "MLOPS_RUNTIME_PARAM_"

    @classmethod
    def get(cls, key):
        runtime_param_key = cls.mangled_param_name(key)
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

    @classmethod
    def mangled_param_name(cls, param_name):
        return f"{cls.PARAM_PREFIX}{param_name}"


class RuntimeParametersLoader:
    """
    This class is used by DRUM to load runtime parameter values from a provided YAML file. It is
    used by DRUM only when executing externally to DataRobot, in a local development environment.

    Here is an example for such `runtime_param_values.yaml` file:
    ```
    PARAM_STR_1: "Example for a string value"
    PARAM_CRED_1:
      credentialType: s3
      awsAccessKeyId: ABDEFGHIJK...
      awsSecretAccessKey: asdjDFSDJafslkjsdDLKGDSDlkjlkj...
      awsSessionToken: null
    """

    def __init__(self, values_filepath):
        if not values_filepath:
            print("Empty runtime parameter values file path!")
            exit(-1)

        try:
            with open(values_filepath, encoding="utf-8") as file:
                try:
                    self._yaml_content = yaml.safe_load(file)
                    if not self._yaml_content:
                        print(f"Runtime parameter values YAML file is empty!")
                        exit(-1)
                except yaml.YAMLError as exc:
                    print(f"Invalid runtime parameter values YAML content! {str(exc)}")
                    exit(-1)
        except FileNotFoundError:
            print(f"Runtime parameter values file does not exist!")
            exit(-1)

    def setup_environment_variables(self):
        credential_payload_trafaret = RuntimeParameterCredentialPayloadTrafaret()
        string_payload_trafaret = RuntimeParameterStringPayloadTrafaret()
        for param_key, param_value in self._yaml_content.items():
            try:
                if isinstance(param_value, dict):  # Only credentials are provided as dict
                    credential_payload = self.credential_attributes_to_underscore(param_value)
                    payload = credential_payload_trafaret.check(
                        {
                            "type": RuntimeParameterTypes.CREDENTIAL.value,
                            "payload": credential_payload,
                        }
                    )
                elif isinstance(param_value, str):  # All remaining supported cases
                    payload = string_payload_trafaret.check(
                        {"type": RuntimeParameterTypes.STRING.value, "payload": param_value}
                    )
            except t.DataError as exc:
                print(f"Failed to load runtime parameter '{param_key}'. {str(exc)}")
                exit(-1)
            mangled_param_key = RuntimeParameters.mangled_param_name(param_key)
            os.environ[mangled_param_key] = json.dumps(payload)

    @classmethod
    def credential_attributes_to_underscore(cls, credential_dict):
        return {cls._camel_to_underscore(key): value for key, value in credential_dict.items()}

    @staticmethod
    def _camel_to_underscore(camel_str):
        return re.sub(r"([A-Z])+([a-z]+)", r"_\1\2", camel_str).lower()
