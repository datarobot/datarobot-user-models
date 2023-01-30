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

from .exceptions import ErrorLoadingRuntimeParameter
from .exceptions import InvalidEmptyYamlContent
from .exceptions import InvalidInputFilePath
from .exceptions import InvalidJsonException
from .exceptions import InvalidRuntimeParam
from .exceptions import InvalidYamlContent
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

    PARAM_PREFIX = "MLOPS_RUNTIME_PARAM"

    # Used to determine if a user has specified a default or not since None is a valid
    # user input.
    _UNSET = object()

    @classmethod
    def get(cls, key, fallback=_UNSET):
        """
        Fetches the value of a runtime parameter as set by the platform. A ValueError is
        raised if the parameter is not set and no fallback argument was provided.

        Parameters
        ----------
        key: str
            The name of the runtime parameter
        fallback: ANY (optional)
            If specified, will be returned if no value has been set by the platform


        Returns
        -------
        The value of the runtime parameter or the fallback (if specified)


        Raises
        ------
        ValueError
            Raised when the parameter key was not set by the platform
        InvalidJsonException
            Raised if there were issues decoding the value of the parameter
        InvalidRuntimeParam
            Raised if the value of the parameter doesn't match the declared type
        """
        runtime_param_key = cls.namespaced_param_name(key)
        if runtime_param_key not in os.environ:
            if fallback is cls._UNSET:
                raise ValueError(
                    f"Runtime parameter '{key}' does not exist and no fallback provided!"
                )
            else:
                return fallback

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
    def namespaced_param_name(cls, param_name):
        return f"{cls.PARAM_PREFIX}_{param_name}"


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
            raise InvalidInputFilePath("Empty runtime parameter values file path!")

        try:
            with open(values_filepath, encoding="utf-8") as file:
                try:
                    self._yaml_content = yaml.safe_load(file)
                    if not self._yaml_content:
                        raise InvalidEmptyYamlContent(
                            "Runtime parameter values YAML file is empty!"
                        )
                except yaml.YAMLError as exc:
                    raise InvalidYamlContent(
                        f"Invalid runtime parameter values YAML content! {str(exc)}"
                    )
        except FileNotFoundError:
            raise InvalidInputFilePath(
                f"Runtime parameter values file does not exist! filepath: {values_filepath}"
            )

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
                raise ErrorLoadingRuntimeParameter(
                    f"Failed to load runtime parameter '{param_key}'. {str(exc)}"
                )
            namespaced_param_key = RuntimeParameters.namespaced_param_name(param_key)
            os.environ[namespaced_param_key] = json.dumps(payload)

    @classmethod
    def credential_attributes_to_underscore(cls, credential_dict):
        return {cls._camel_to_underscore(key): value for key, value in credential_dict.items()}

    @staticmethod
    def _camel_to_underscore(camel_str):
        return re.sub(r"([A-Z])+([a-z]+)", r"_\1\2", camel_str).lower()
