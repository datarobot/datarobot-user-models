"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml
from datarobot_drum.drum.drum import CMRunner

from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME, ArgumentsOptions, ArgumentOptionsEnvVars
from datarobot_drum.drum.enum import ModelMetadataKeys
from datarobot_drum.runtime_parameters.exceptions import (
    InvalidEmptyYamlContent,
    ErrorLoadingRuntimeParameter,
)
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
            (RuntimeParameterTypes.BOOLEAN, True),
            (RuntimeParameterTypes.NUMERIC, 10),
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
            ("CREDENTIAL", {"credentialType": "s3"}),
            ("creds", {"credentialType": "s3"}),
        ],
    )
    def test_invalid_credential_type(self, runtime_param_type, payload):
        self._read_runtime_param_and_expect_to_fail(runtime_param_type, payload)

    @pytest.mark.parametrize("payload", ["string-payload", {"credentialType": "s3"}])
    def test_invalid_boolean_type(self, payload):
        self._read_runtime_param_and_expect_to_fail(type="boolean", payload=payload)

    @pytest.mark.parametrize("payload", ["string-payload", {"credentialType": "s3"}])
    def test_invalid_numeric_type(self, payload):
        self._read_runtime_param_and_expect_to_fail(type="numeric", payload=payload)

    @staticmethod
    def _read_runtime_param_and_expect_to_fail(type, payload):
        runtime_param_name = "AAA"
        namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name(runtime_param_name)
        runtime_param_env_value = json.dumps({"type": type, "payload": payload})
        with patch.dict(os.environ, {namespaced_runtime_param_name: runtime_param_env_value}):
            with pytest.raises(InvalidRuntimeParam, match=r".*Invalid runtime parameter!.*"):
                RuntimeParameters.get(runtime_param_name)

    def test_credential_missing_credential_type(self):
        payload = {
            "credentialType": "s3",
            "region": "us-west",
            "awsAccessKeyId": "123aaa",
            "awsSecretAccessKey": "3425sdd",
            "awsSessionToken": "123aaa",
        }
        required = "credentialType"
        payload.pop(required)
        self._read_runtime_param_and_expect_to_fail(RuntimeParameterTypes.CREDENTIAL.value, payload)

    def test_credential_empty_credential_type(self):
        payload = {
            "credentialType": "",
            "region": "us-west",
            "awsAccessKeyId": "123aaa",
            "awsSecretAccessKey": "3425sdd",
            "awsSessionToken": "123aaa",
        }
        self._read_runtime_param_and_expect_to_fail(RuntimeParameterTypes.CREDENTIAL.value, payload)

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
            "STR_PARAM1": "Some value",
            "BOOL_PARAM1": True,
            "NUMERIC_PARAM1": 50,
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
            {"fieldName": "STR_PARAM1", "type": "string", "defaultValue": "Hello world!"},
            {"fieldName": "STR_PARAM2", "type": "string", "defaultValue": "goodbye"},
            {"fieldName": "BOOL_PARAM", "type": "boolean", "defaultValue": False},
            {
                "fieldName": "NUMERIC_PARAM1",
                "type": "numeric",
                "defaultValue": 50,
                "minValue": 0,
                "maxValue": 100,
            },
            {"fieldName": "MANDATORY_STR_PARAM", "type": "string", "allowEmpty": False},
            {"fieldName": "S3_CRED_PARAM", "type": "credential", "description": "a secret"},
            {"fieldName": "OTHER_CRED_PARAM", "type": "credential", "defaultValue": None},
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
    def model_metadata_duplicate_definitions_file(self, empty_code_dir):
        f = empty_code_dir / MODEL_CONFIG_FILENAME
        content = {
            ModelMetadataKeys.NAME: "model name",
            ModelMetadataKeys.TYPE: "inference",
            ModelMetadataKeys.TARGET_TYPE: "regression",
            ModelMetadataKeys.RUNTIME_PARAMETERS: [
                {"fieldName": "STR_PARAM1", "type": "string", "defaultValue": "Hello world!"},
                {"fieldName": "STR_PARAM1", "type": "string", "defaultValue": "goodbye"},
            ],
        }
        f.write_text(yaml.dump(content))
        return f

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

    def test_none_values_path(self, model_metadata_file):
        with pytest.raises(InvalidInputFilePath, match="Empty runtime parameter values file path!"):
            RuntimeParametersLoader(None, model_metadata_file.parent)

    def test_none_metadata(self, runtime_params_values_file):
        with pytest.raises(InvalidInputFilePath, match="Empty code-dir path!"):
            RuntimeParametersLoader(runtime_params_values_file, None)

    def test_values_file_not_exists(self, runtime_params_values_file, model_metadata_file):
        runtime_params_values_file.unlink()
        with pytest.raises(
            InvalidInputFilePath, match="Runtime parameter values file does not exist!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_metadata_file_not_exists(self, runtime_params_values_file, empty_code_dir):
        with pytest.raises(InvalidInputFilePath, match="must exist to use runtime parameters"):
            RuntimeParametersLoader(runtime_params_values_file, empty_code_dir)

    def test_empty_values_file(self, runtime_params_values_file, model_metadata_file):
        runtime_params_values_file.write_text("")
        with pytest.raises(
            InvalidEmptyYamlContent, match="Runtime parameter values YAML file is empty!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_empty_metadata_file(self, runtime_params_values_file, model_metadata_file):
        model_metadata_file.write_text("")
        with pytest.raises(InvalidEmptyYamlContent, match="Model-metadata YAML file is empty!"):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_no_defs_metadata_file(self, runtime_params_values_file, model_metadata_file):
        model_metadata_file.write_text("name: my model")
        with pytest.raises(InvalidYamlContent, match="must contain at least one parameter def"):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_invalid_values_yaml_content(self, runtime_params_values_file, model_metadata_file):
        invalid_yaml_content = "['something': 1"
        runtime_params_values_file.write_text(invalid_yaml_content)
        with pytest.raises(
            InvalidYamlContent, match="Invalid runtime parameter values YAML content!"
        ):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_invalid_metadata_yaml_content(self, runtime_params_values_file, model_metadata_file):
        invalid_yaml_content = "['something': 1"
        model_metadata_file.write_text(invalid_yaml_content)
        with pytest.raises(InvalidYamlContent, match="Invalid model-metadata YAML content!"):
            RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)

    def test_invalid_numeric_parameter_values(
        self, runtime_params_values_file, model_metadata_file
    ):
        invalid_numeric_var_yaml_content = {"NUMERIC_PARAM1": 500}
        runtime_params_values_file.write_text(yaml.dump(invalid_numeric_var_yaml_content))
        with pytest.raises(ErrorLoadingRuntimeParameter, match="value is greater than 100"):
            loader = RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)
            loader.setup_environment_variables()

    def test_duplicate_definitions(
        self, runtime_params_values_file, model_metadata_duplicate_definitions_file
    ):
        with pytest.raises(ErrorLoadingRuntimeParameter, match="duplicated definition"):
            RuntimeParametersLoader(
                runtime_params_values_file, model_metadata_duplicate_definitions_file.parent
            )

    def test_setup_success(
        self,
        runtime_parameter_values,
        runtime_params_values_file,
        runtime_parameter_definitions,
        model_metadata_file,
    ):
        loader = RuntimeParametersLoader(runtime_params_values_file, model_metadata_file.parent)
        with patch.dict("os.environ"):
            loader.setup_environment_variables()

            for param_def in runtime_parameter_definitions:
                param_name = param_def["fieldName"]
                default = param_def.get("defaultValue")
                override_value = runtime_parameter_values.get(param_name)
                expected_value = override_value if override_value is not None else default

                actual_value = RuntimeParameters.get(param_name)
                assert actual_value == expected_value


class TestRuntimeParametersDockerCommand:
    @pytest.fixture
    def tmp_dir(self):
        with TemporaryDirectory(suffix="output-dir") as dir_name:
            yield dir_name

    @pytest.fixture
    def set_runtime_param_file_env_var(self):
        def _f(file_path):
            os.environ[ArgumentOptionsEnvVars.RUNTIME_PARAMS_FILE] = file_path

        yield _f
        os.environ.pop(ArgumentOptionsEnvVars.RUNTIME_PARAMS_FILE, None)

    @pytest.fixture
    def fit_args(self, tmp_dir):
        return [
            "fit",
            "--code-dir",
            tmp_dir,
            "--input",
            __file__,
            "--target",
            "some_target",
            "--target-type",
            "regression",
            "--output",
            tmp_dir,
        ]

    @pytest.fixture
    def score_args(self, tmp_dir):
        return [
            "score",
            "--code-dir",
            tmp_dir,
            "--input",
            __file__,
            "--target-type",
            "regression",
            "--language",
            "python",
        ]

    @pytest.fixture
    def server_args(self, tmp_dir):
        return [
            "server",
            "--code-dir",
            tmp_dir,
            "--address",
            "allthedice.com:1234",
            "--target-type",
            "regression",
            "--language",
            "python",
        ]

    def test_runtime_params_invalid(self, fit_args, runtime_factory):
        with pytest.raises(SystemExit), NamedTemporaryFile() as f:
            fit_args.extend([ArgumentsOptions.RUNTIME_PARAMS_FILE, f.name])
            runtime_factory(fit_args)

    @pytest.mark.parametrize("args", ["server_args", "score_args"])
    @pytest.mark.parametrize("add_runtime_param_file", [True, False])
    def test_runtime_params_as_parameter(
        self, request, args, runtime_factory, add_runtime_param_file
    ):
        args_list = request.getfixturevalue(args)

        with (
            patch.object(CMRunner, "_maybe_build_image"),
            NamedTemporaryFile(suffix="_runtime.yaml") as runtime_param_file,
        ):
            # We add the runtime parameter file as parameter
            params_file_host_location = os.path.realpath(runtime_param_file.name)
            if add_runtime_param_file:
                args_list.extend([ArgumentsOptions.RUNTIME_PARAMS_FILE, params_file_host_location])

            cm_runner = runtime_factory(args_list)
            docker_cmd = cm_runner._prepare_docker_command(
                cm_runner.options, cm_runner.run_mode, cm_runner.raw_arguments
            )
            assert docker_cmd
            assert params_file_host_location not in docker_cmd

            params_file_docker_location = "/opt/runtime_parameters.yaml"
            file_mapping_param = f"{params_file_host_location}:{params_file_docker_location}"
            if add_runtime_param_file:
                assert file_mapping_param in docker_cmd
                assert params_file_docker_location in docker_cmd
            else:
                assert file_mapping_param not in docker_cmd
                assert params_file_docker_location not in docker_cmd

    @pytest.mark.parametrize("args", ["server_args", "score_args"])
    @pytest.mark.parametrize("add_var_runtime_param_file", [True, False])
    def test_runtime_params_as_env_var(
        self,
        request,
        args,
        runtime_factory,
        add_var_runtime_param_file,
        set_runtime_param_file_env_var,
    ):
        args_list = request.getfixturevalue(args)

        with patch.object(CMRunner, "_maybe_build_image"), NamedTemporaryFile(
            suffix="_runtime.yaml"
        ) as runtime_param_file:
            # We add the runtime parameter file as parameter
            params_file_host_location = os.path.realpath(runtime_param_file.name)
            params_file_docker_location = "/opt/runtime_parameters.yaml"

            if add_var_runtime_param_file:
                set_runtime_param_file_env_var(params_file_host_location)

            cm_runner = runtime_factory(args_list)
            docker_cmd = cm_runner._prepare_docker_command(
                cm_runner.options, cm_runner.run_mode, cm_runner.raw_arguments
            )
            assert docker_cmd
            assert params_file_host_location not in docker_cmd
            assert params_file_docker_location not in docker_cmd

            file_mapping_param = f"{params_file_host_location}:{params_file_docker_location}"
            var_mapping_param = (
                f"{ArgumentOptionsEnvVars.RUNTIME_PARAMS_FILE}={params_file_docker_location}"
            )
            if add_var_runtime_param_file:
                assert file_mapping_param in docker_cmd
                assert var_mapping_param in docker_cmd
            else:
                assert file_mapping_param not in docker_cmd
                assert var_mapping_param not in docker_cmd
