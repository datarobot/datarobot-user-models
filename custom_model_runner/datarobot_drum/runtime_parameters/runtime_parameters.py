"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from collections import namedtuple

import trafaret as t
import yaml

from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.enum import ModelMetadataKeys
from datarobot_drum.runtime_parameters.exceptions import ErrorLoadingRuntimeParameter
from datarobot_drum.runtime_parameters.exceptions import InvalidEmptyYamlContent
from datarobot_drum.runtime_parameters.exceptions import InvalidInputFilePath
from datarobot_drum.runtime_parameters.exceptions import InvalidJsonException
from datarobot_drum.runtime_parameters.exceptions import InvalidRuntimeParam
from datarobot_drum.runtime_parameters.exceptions import InvalidYamlContent
from datarobot_drum.runtime_parameters.runtime_parameters_schema import (
    RuntimeParameterCredentialPayloadTrafaret,
    RuntimeParameterDeploymentPayloadTrafaret,
    RuntimeParameterBooleanPayloadTrafaret,
    RuntimeParameterNumericPayloadTrafaret,
)
from datarobot_drum.runtime_parameters.runtime_parameters_schema import (
    RuntimeParameterPayloadTrafaret,
)
from datarobot_drum.runtime_parameters.runtime_parameters_schema import (
    RuntimeParameterStringPayloadTrafaret,
)
from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes
from datarobot_drum.runtime_parameters.runtime_parameters_schema import (
    RuntimeParameterDefinitionTrafaret,
)


class RuntimeParameters:
    """
    A class that is used to read runtime-parameters that are delivered to the executed
    custom model. The runtime parameters are supposed to be defined by the user via the DataRobot
    web UI. Do not try to bypass this class by writing your own proprietary code, because the
    internal implementation may change over time.
    """

    PARAM_PREFIX = "MLOPS_RUNTIME_PARAM"

    @classmethod
    def get(cls, key):
        """
        Fetches the value of a runtime parameter as set by the platform. A ValueError is
        raised if the parameter is not set.

        Parameters
        ----------
        key: str
            The name of the runtime parameter


        Returns
        -------
        The value of the runtime parameter


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
    def namespaced_param_name(cls, param_name):
        return f"{cls.PARAM_PREFIX}_{param_name}"

    @classmethod
    def has(cls, param_name):
        runtime_param_key = cls.namespaced_param_name(param_name)
        return runtime_param_key in os.environ


class RuntimeParametersLoader:
    """
    This class is used by DRUM to load runtime parameter values from a provided YAML file. It is
    used by DRUM only when executing externally to DataRobot (i.e. in a local development
    environment).

    Here is an example for such `runtime_param_values.yaml` file:
    ```
    PARAM_STR_1: "Example for a string value"
    PARAM_CRED_1:
      credentialType: s3
      awsAccessKeyId: ABDEFGHIJK...
      awsSecretAccessKey: asdjDFSDJafslkjsdDLKGDSDlkjlkj...
      awsSessionToken: null
    ```
    """

    ParameterDefinition = namedtuple(
        "ParameterDefinition", ["name", "type", "default", "allow_empty", "min_value", "max_value"]
    )

    def __init__(self, values_filepath, code_dir):
        if not values_filepath:
            raise InvalidInputFilePath("Empty runtime parameter values file path!")
        if not code_dir:
            raise InvalidInputFilePath("Empty code-dir path!")

        self._load_parameter_definitions(code_dir)

        try:
            with open(values_filepath, encoding="utf-8") as file:
                self._yaml_content = yaml.safe_load(file)
            if not self._yaml_content:
                raise InvalidEmptyYamlContent("Runtime parameter values YAML file is empty!")
        except yaml.YAMLError as exc:
            raise InvalidYamlContent(f"Invalid runtime parameter values YAML content! {str(exc)}")
        except FileNotFoundError:
            raise InvalidInputFilePath(
                f"Runtime parameter values file does not exist! filepath: {values_filepath}"
            )

    def _load_parameter_definitions(self, code_dir):
        try:
            with open(os.path.join(code_dir, MODEL_CONFIG_FILENAME)) as file:
                model_metadata = yaml.safe_load(file)
            if not model_metadata:
                raise InvalidEmptyYamlContent("Model-metadata YAML file is empty!")
        except yaml.YAMLError as exc:
            raise InvalidYamlContent(f"Invalid model-metadata YAML content! {str(exc)}")
        except FileNotFoundError:
            raise InvalidInputFilePath(
                f"{MODEL_CONFIG_FILENAME} must exist to use runtime parameters"
            )

        parameters = model_metadata.get(ModelMetadataKeys.RUNTIME_PARAMETERS)
        if not parameters:
            raise InvalidYamlContent(
                f"{MODEL_CONFIG_FILENAME}: YAML file must contain at least one "
                f"parameter definition in the section '{ModelMetadataKeys.RUNTIME_PARAMETERS}'"
            )
        self._parameter_definitions = {}
        for parameter in parameters:
            try:
                data = RuntimeParameterDefinitionTrafaret.check(parameter)
            except t.DataError as exc:
                raise ErrorLoadingRuntimeParameter(f"Failed to load runtime parameter: {str(exc)}")

            # Check duplicate definitions
            param_name = data["name"]
            if param_name in self._parameter_definitions:
                raise ErrorLoadingRuntimeParameter(
                    f"Failed to load runtime parameter [{param_name}], duplicated definition"
                )
            self._parameter_definitions[param_name] = self.ParameterDefinition(**data)

    def setup_environment_variables(self):
        credential_payload_trafaret = RuntimeParameterCredentialPayloadTrafaret()
        string_payload_trafaret = RuntimeParameterStringPayloadTrafaret()
        boolean_payload_trafaret = RuntimeParameterBooleanPayloadTrafaret()
        deployment_payload_trafaret = RuntimeParameterDeploymentPayloadTrafaret()
        for param_key, param_definition in self._parameter_definitions.items():
            param_value = self._yaml_content.get(param_key, param_definition.default)

            try:
                if param_definition.type == RuntimeParameterTypes.CREDENTIAL:
                    payload = credential_payload_trafaret.check(
                        {
                            "type": RuntimeParameterTypes.CREDENTIAL.value,
                            "payload": param_value,
                        }
                    )
                elif param_definition.type == RuntimeParameterTypes.STRING:
                    payload = string_payload_trafaret.check(
                        {"type": RuntimeParameterTypes.STRING.value, "payload": param_value}
                    )
                elif param_definition.type == RuntimeParameterTypes.BOOLEAN:
                    payload = boolean_payload_trafaret.check(
                        {"type": RuntimeParameterTypes.BOOLEAN.value, "payload": param_value}
                    )
                elif param_definition.type == RuntimeParameterTypes.DEPLOYMENT:
                    payload = deployment_payload_trafaret.check(
                        {"type": RuntimeParameterTypes.DEPLOYMENT.value, "payload": param_value}
                    )
                elif param_definition.type == RuntimeParameterTypes.NUMERIC:
                    numeric_payload_trafaret = RuntimeParameterNumericPayloadTrafaret(
                        min_value=param_definition.min_value,
                        max_value=param_definition.max_value,
                    )

                    payload = numeric_payload_trafaret.check(
                        {"type": RuntimeParameterTypes.NUMERIC.value, "payload": param_value}
                    )
                else:
                    raise ErrorLoadingRuntimeParameter(
                        f"Unsupported type '{param_definition.type} for parameter {param_key}"
                    )
            except t.DataError as exc:
                raise ErrorLoadingRuntimeParameter(
                    f"Failed to load runtime parameter '{param_key}'. {str(exc)}"
                )
            namespaced_param_key = RuntimeParameters.namespaced_param_name(param_key)
            os.environ[namespaced_param_key] = json.dumps(payload)
