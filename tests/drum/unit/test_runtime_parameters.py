"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from unittest.mock import patch

import pytest
import yaml

from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.enum import ModelMetadataKeys
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
                    "credentialType": "s3",
                    "awsAccessKeyId": "123aaa",
                    "awsSecretAccessKey": "3425sdd",
                    "awsSessionToken": "12345abcde",
                },
            ),
            (
                RuntimeParameterTypes.CREDENTIAL,
                {
                    "credentialType": "s3",
                    "region": "us-west",
                    "awsAccessKeyId": "123aaa",
                    "awsSecretAccessKey": "3425sdd",
                    "awsSessionToken": "12345abcde",
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
                    "credentialType": "s3",
                    "awsAccessKeyId": "123aaa",
                    "awsSecretAccessKey": "3425sdd",
                    "awsSessionToken": "12345abcde",
                },
            ),
            (
                "creds",
                {
                    "credentialType": "s3",
                    "awsAccessKeyId": "123aaa",
                    "awsSecretAccessKey": "3425sdd",
                    "awsSessionToken": "12345abcde",
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
            "credentialType": "s3",
            "region": "us-west",
            "awsAccessKeyId": "123aaa",
            "awsSecretAccessKey": "3425sdd",
            "awsSessionToken": "123aaa",
        }
        for missing_attr in (
            "credentialType",
            "awsAccessKeyId",
            "awsSecretAccessKey",
            "awsSessionToken",
        ):
            payload.pop(missing_attr)
            self._read_runtime_param_and_expect_to_fail(
                RuntimeParameterTypes.CREDENTIAL.value, payload
            )

    def test_empty_mandatory_aws_credential_attribute(self):
        payload = {
            "credentialType": "s3",
            "region": "us-west",
            "awsAccessKeyId": "123aaa",
            "awsSecretAccessKey": "3425sdd",
            "awsSessionToken": "123aaa",
        }
        for missing_attr in (
            "credentialType",
            "awsAccessKeyId",
            "awsSecretAccessKey",
            "awsSessionToken",
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
    @pytest.fixture
    def runtime_parameter_values(self):
        return {
            "STR_PARAM": "Some value",
            "S3_CRED_PARAM": {
                "credentialType": "s3",
                "awsAccessKeyId": "AWOUIEOIUI",
                "awsSecretAccessKey": "SDFLKJAlskjflsjfsLKDKJF",
                "awsSessionToken": None,
            },
        }

    @pytest.fixture
    def runtime_parameter_definitions(self):
        return [
            {"fieldName": "STR_PARAM", "type": "string", "defaultValue": "Hello world!"},
            {"fieldName": "S3_CRED_PARAM", "type": "credential", "description": "a secret"},
        ]

    @pytest.fixture
    def runtime_params_values_file(self, tmp_path, runtime_parameter_values):
        p = tmp_path / ".runtime_param_values.yaml"
        p.write_text(yaml.dump(runtime_parameter_values))
        return p

    @pytest.fixture
    def empty_code_dir(self, tmp_path):
        p = tmp_path / "model-data"
        p.mkdir()
        return p

    @pytest.fixture
    def model_metadata_file(self, empty_code_dir, runtime_parameter_definitions):
        f = empty_code_dir / MODEL_CONFIG_FILENAME
        content = {
            ModelMetadataKeys.NAME: "model name",
            ModelMetadataKeys.TYPE: "inference",
            ModelMetadataKeys.TARGET_TYPE: "regression",
            ModelMetadataKeys.RUNTIME_PARAMETERS: runtime_parameter_definitions,
        }
        f.write_text(yaml.dump(content))
        return f

    def test_none_filepath(self, model_metadata_file):
        with pytest.raises(InvalidInputFilePath, match="Empty runtime parameter values file path!"):
            RuntimeParametersLoader(None, model_metadata_file.parent)

    def test_none_metadata(self, runtime_params_values_file):
        with pytest.raises(InvalidInputFilePath, match="Empty code-dir path!"):
            RuntimeParametersLoader(runtime_params_values_file, None)

    def test_file_not_exists(self, runtime_params_values_file, model_metadata_file):
        runtime_params_values_file.unlink()
        with pytest.raises(
            InvalidInputFilePath, match="Runtime parameter values file does not exist!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_metadata_file_not_exists(self, runtime_params_values_file, empty_code_dir):
        with pytest.raises(InvalidInputFilePath, match="must exist to use runtime parameters"):
            RuntimeParametersLoader(runtime_params_values_file, empty_code_dir)

    def test_empty_file(self, runtime_params_values_file, model_metadata_file):
        runtime_params_values_file.write_text("")
        with pytest.raises(
            InvalidEmptyYamlContent, match="Runtime parameter values YAML file is empty!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_invalid_yaml_content(self, runtime_params_values_file, model_metadata_file):
        invalid_yaml_content = "['something': 1"
        runtime_params_values_file.write_text(invalid_yaml_content)
        with pytest.raises(
            InvalidYamlContent, match="Invalid runtime parameter values YAML content!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_setup_success(
        self, runtime_parameter_values, runtime_params_values_file, model_metadata_file
    ):
        try:
            loader = RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)
            loader.setup_environment_variables()

            for param_name, _ in runtime_parameter_values.items():
                actual_value = RuntimeParameters.get(param_name)
                expected_value = runtime_parameter_values[param_name]
                assert actual_value == expected_value
        finally:
            for param_name, _ in runtime_parameter_values.items():
                os.environ.pop(RuntimeParameters.namespaced_param_name(param_name))
