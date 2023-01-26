"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import contextlib
import json
import os
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
import yaml

from datarobot_drum.runtime_parameters.exceptions import InvalidEmptyYamlContent
from datarobot_drum.runtime_parameters.exceptions import InvalidInputFilePath
from datarobot_drum.runtime_parameters.exceptions import InvalidJsonException
from datarobot_drum.runtime_parameters.exceptions import InvalidRuntimeParam
from datarobot_drum.runtime_parameters.exceptions import InvalidYamlContent
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParametersLoader
from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes


class TestRuntimeParameters:
    @pytest.mark.parametrize(
        "runtime_param_type, payload",
        [
            (RuntimeParameterTypes.STRING, "Some string value"),
            (
                RuntimeParameterTypes.CREDENTIAL,
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
            (
                RuntimeParameterTypes.CREDENTIAL,
                {
                    "credential_type": "s3",
                    "region": "us-west",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
        ],
    )
    def test_valid(self, runtime_param_type, payload):
        runtime_param_name = "AAA"
        namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name(runtime_param_name)
        runtime_param_env_value = json.dumps({"type": runtime_param_type.value, "payload": payload})
        with patch.dict(os.environ, {namespaced_runtime_param_name: runtime_param_env_value}):
            assert RuntimeParameters.get(runtime_param_name) == payload

    @pytest.mark.parametrize(
        "runtime_param_type, payload",
        [
            ("STRING", "Some string value"),
            ("str", "Some string value"),
            (
                "CREDENTIAL",
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
            (
                "creds",
                {
                    "credential_type": "s3",
                    "aws_access_key_id": "123aaa",
                    "aws_secret_access_key": "3425sdd",
                    "aws_session_token": "12345abcde",
                },
            ),
        ],
    )
    def test_invalid_type(self, runtime_param_type, payload):
        self._read_runtime_param_and_expect_to_fail(runtime_param_type, payload)

    @staticmethod
    def _read_runtime_param_and_expect_to_fail(type, payload):
        runtime_param_name = "AAA"
        namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name(runtime_param_name)
        runtime_param_env_value = json.dumps({"type": type, "payload": payload})
        with patch.dict(os.environ, {namespaced_runtime_param_name: runtime_param_env_value}):
            with pytest.raises(InvalidRuntimeParam, match=r".*Invalid runtime parameter!.*"):
                RuntimeParameters.get(runtime_param_name)

    def test_missing_mandatory_aws_credential_attribute(self):
        payload = {
            "credential_type": "s3",
            "region": "us-west",
            "aws_access_key_id": "123aaa",
            "aws_secret_access_key": "3425sdd",
            "aws_session_token": "123aaa",
        }
        for missing_attr in (
            "credential_type",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ):
            payload.pop(missing_attr)
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_empty_mandatory_aws_credential_attribute(self):
        payload = {
            "credential_type": "s3",
            "region": "us-west",
            "aws_access_key_id": "123aaa",
            "aws_secret_access_key": "3425sdd",
            "aws_session_token": "123aaa",
        }
        for missing_attr in (
            "credential_type",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ):
            payload[missing_attr] = ""
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_invalid_json_env_value(self):
        runtime_param_name = "AAA"
        namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name(runtime_param_name)
        runtime_param_value = json.dumps(
            {"type": RuntimeParameterTypes.STRING.value, "payload": "abc"}
        )
        invalid_env_value = runtime_param_value + " - invalid"
        with patch.dict(os.environ, {namespaced_runtime_param_name: invalid_env_value}):
            with pytest.raises(
                InvalidJsonException, match=r".*Invalid runtime parameter json payload.*"
            ):
                RuntimeParameters.get(runtime_param_name)


class TestRuntimeParametersLoader:
    def test_none_filepath(self):
        with pytest.raises(InvalidInputFilePath) as exc:
            RuntimeParametersLoader(None)
        assert "Empty runtime parameter values file path!" in str(exc.value)

    def test_file_not_exists(self):
        with pytest.raises(InvalidInputFilePath) as exc:
            RuntimeParametersLoader("/tmp/non-existing-12er.yaml")
        assert "Runtime parameter values file does not exist!" in str(exc.value)

    def test_empty_file(self):
        with NamedTemporaryFile("w", encoding="utf-8") as file:
            with pytest.raises(InvalidEmptyYamlContent) as exc:
                RuntimeParametersLoader(file.name)
            assert "Runtime parameter values YAML file is empty!" in str(exc.value)

    @contextlib.contextmanager
    def _runtime_params_yaml_file(self, yaml_content):
        with NamedTemporaryFile("w", encoding="utf-8") as file:
            file.write(yaml_content)
            file.flush()
            yield file.name

    def test_invalid_yaml_content(self):
        valid_yaml_content = yaml.dump({"PARAM_STR": "Some value"})
        invalid_yaml_content = f"[{valid_yaml_content}"
        with self._runtime_params_yaml_file(invalid_yaml_content) as filepath:
            with pytest.raises(InvalidYamlContent) as exc:
                RuntimeParametersLoader(filepath)
            assert "Invalid runtime parameter values YAML content!" in str(exc.value)

    def test_setup_success(self):
        runtime_parameter_values = {
            "STR_PARAM": "Some value",
            "S3_CRED_PARAM": {
                "credentialType": "s3",
                "awsAccessKeyId": "AWOUIEOIUI",
                "awsSecretAccessKey": "SDFLKJAlskjflsjfsLKDKJF",
                "awsSessionToken": None,
            },
        }
        try:
            valid_yaml_content = yaml.dump(runtime_parameter_values)
            with self._runtime_params_yaml_file(valid_yaml_content) as filepath:
                RuntimeParametersLoader(filepath).setup_environment_variables()

            for param_name, param_value in runtime_parameter_values.items():
                actual_value = RuntimeParameters.get(param_name)
                expected_value = runtime_parameter_values[param_name]
                if isinstance(expected_value, dict):
                    expected_value = RuntimeParametersLoader.credential_attributes_to_underscore(
                        runtime_parameter_values[param_name]
                    )
                assert actual_value == expected_value
        finally:
            for param_name, param_value in runtime_parameter_values.items():
                os.environ.pop(RuntimeParameters.namespaced_param_name(param_name))
