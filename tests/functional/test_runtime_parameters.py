import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
import yaml

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.enum import ModelMetadataKeys

from datarobot_drum.resource.utils import _create_custom_model_dir
from datarobot_drum.resource.utils import _exec_shell_cmd
from tests.constants import PYTHON_UNSTRUCTURED_RUNTIME_PARAMS
from tests.constants import UNSTRUCTURED
from tests.fixtures.unstructured_custom_runtime_parameters import EXPECTED_RUNTIME_PARAMS_FILE_NAME


def _setup_expected_runtime_parameters(
    custom_model_dir, is_missing_attr, bool_var_value, numeric_var_value
):
    expected_runtime_params = {
        "SOME_STR_KEY": {"type": "string", "payload": "Hello"},
        "SOME_AWS_CRED_KEY": {
            "type": "credential",
            "payload": {
                "credentialType": "s3",
                "region": "us-west",
                "awsAccessKeyId": "123",
                "awsSecretAccessKey": "abc",
                "awsSessionToken": "456edf",
            },
        },
        "SOME_DEPLOYMENT_KEY": {"type": "deployment", "payload": "65415890b9b0fd93778e6935"},
        "SOME_BOOLEAN_KEY": {"type": "boolean", "payload": bool_var_value},
        "SOME_NUMERIC_KEY": {"type": "numeric", "payload": numeric_var_value},
    }
    if is_missing_attr:
        expected_runtime_params["SOME_AWS_CRED_KEY"]["payload"].pop("credentialType")

    model_dir = Path(custom_model_dir)
    expected_runtime_param_filepath = model_dir / EXPECTED_RUNTIME_PARAMS_FILE_NAME
    # Need to dump as JSON because custom.py for this test expects it
    expected_runtime_param_filepath.write_text(json.dumps(expected_runtime_params))

    expected_runtime_def_filepath = model_dir / MODEL_CONFIG_FILENAME
    expected_runtime_def_filepath.write_text(
        # Need to dump as YAML because strictyaml doesn't support JSONesque style
        yaml.dump(
            {
                ModelMetadataKeys.NAME: "my model",
                ModelMetadataKeys.TYPE: "inference",
                ModelMetadataKeys.TARGET_TYPE: "unstructured",
                ModelMetadataKeys.RUNTIME_PARAMETERS: [
                    dict(fieldName=k, type=v["type"]) for k, v in expected_runtime_params.items()
                ],
            },
        )
    )
    return expected_runtime_params, expected_runtime_param_filepath


class TestRuntimeParametersFromEnv:
    def test_runtime_parameters_success(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, bool_var_value=True
        )
        assert not stderr

    def _test_custom_model_with_runtime_params(
        self,
        resources,
        tmp_path,
        is_invalid_json=False,
        is_missing_attr=False,
        bool_var_value=False,
        numeric_var_value=123,
    ):
        problem = UNSTRUCTURED
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework=None,
            problem=problem,
            language=PYTHON_UNSTRUCTURED_RUNTIME_PARAMS,
        )

        runtime_params_env_values, runtime_param_filepath = self._setup_runtime_parameters(
            custom_model_dir, is_invalid_json, is_missing_attr, bool_var_value, numeric_var_value
        )

        cmd = (
            f"{ArgumentsOptions.MAIN_COMMAND} score "
            f"--code-dir {custom_model_dir} "
            f"--input {runtime_param_filepath} "
            f"--output {Path(tmp_path) / 'output'} "
            f"--target-type {resources.target_types(problem)}"
        )

        with patch.dict(os.environ, runtime_params_env_values):
            _, _, stderr = _exec_shell_cmd(cmd, err_msg=None, assert_if_fail=False)
        return stderr

    @classmethod
    def _setup_runtime_parameters(
        cls, custom_model_dir, is_invalid_json, is_missing_attr, bool_var_value, numeric_var_value
    ):
        runtime_params, runtime_params_filepath = _setup_expected_runtime_parameters(
            custom_model_dir, is_missing_attr, bool_var_value, numeric_var_value
        )

        expected_runtime_params_env_value = {
            f"MLOPS_RUNTIME_PARAM_{k}": json.dumps(v) for k, v in runtime_params.items()
        }
        if is_invalid_json:
            for k in expected_runtime_params_env_value.keys():
                expected_runtime_params_env_value[k] += "-invalid"

        return expected_runtime_params_env_value, runtime_params_filepath

    def test_runtime_parameters_invalid_json(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, is_invalid_json=True
        )
        assert "Invalid runtime parameter json payload." in stderr

    def test_runtime_parameters_missing_attr(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, is_missing_attr=True
        )
        assert re.search(
            r".*Invalid runtime parameter!.*{\\\\\\\\\\\\\\\'credentialType\\\\\\\\\\\\\\\': "
            r"DataError\(\\\\\\\\\\\\\\\'is required\\\\\\\\\\\\\\\'\)}.*",
            stderr,
        )

    def test_runtime_parameters_boolean_invalid(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, bool_var_value="text"
        )
        assert re.search(
            r".*Invalid runtime parameter!.*value should be True or False.*",
            stderr,
        )

    def test_runtime_parameters_numeric_invalid(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, numeric_var_value="text"
        )
        assert re.search(
            r".*Invalid runtime parameter!.*value can.*'t be converted to float.*",
            stderr,
        )


class TestRuntimeParametersFromValuesFile:
    @pytest.fixture
    def runtime_param_values_stream(self):
        with NamedTemporaryFile() as file_stream:
            yield file_stream

    @pytest.mark.parametrize("use_runtime_params_env_var", [True, False])
    def test_runtime_parameters_success(
        self, resources, tmp_path, runtime_param_values_stream, use_runtime_params_env_var
    ):
        stderr = self._test_custom_model_with_runtime_params(
            resources,
            tmp_path,
            runtime_param_values_stream,
            use_runtime_params_env_var=use_runtime_params_env_var,
        )
        assert not stderr

    def _test_custom_model_with_runtime_params(
        self,
        resources,
        tmp_path,
        runtime_param_values_stream,
        is_invalid_yaml=False,
        is_missing_attr=False,
        bool_var_value=False,
        numeric_var_value=123,
        use_runtime_params_env_var=False,
    ):
        problem = UNSTRUCTURED
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework=None,
            problem=problem,
            language=PYTHON_UNSTRUCTURED_RUNTIME_PARAMS,
        )

        runtime_params_filepath = self._setup_runtime_parameters(
            custom_model_dir,
            runtime_param_values_stream,
            is_invalid_yaml=is_invalid_yaml,
            is_missing_attr=is_missing_attr,
            bool_var_value=bool_var_value,
            numeric_var_value=numeric_var_value,
        )

        cmd = (
            f"{ArgumentsOptions.MAIN_COMMAND} score "
            f"--code-dir {custom_model_dir} "
            f"--input {runtime_params_filepath} "
            f"--output {Path(tmp_path) / 'output'} "
            f"--target-type {resources.target_types(problem)} "
        )

        env = os.environ
        if use_runtime_params_env_var:
            cmd += f"--runtime-params-file {runtime_param_values_stream.name}"
        else:
            env["RUNTIME_PARAMS_FILE"] = runtime_param_values_stream.name
        _, stdout, _ = _exec_shell_cmd(cmd, err_msg=None, assert_if_fail=False, env=env)
        return stdout

    @classmethod
    def _setup_runtime_parameters(
        cls,
        custom_model_dir,
        runtime_param_values_stream,
        is_invalid_yaml,
        is_missing_attr,
        bool_var_value,
        numeric_var_value,
    ):
        runtime_params, runtime_params_filepath = _setup_expected_runtime_parameters(
            custom_model_dir, is_missing_attr, bool_var_value, numeric_var_value
        )

        yaml_content = yaml.dump({key: value["payload"] for key, value in runtime_params.items()})
        if is_invalid_yaml:
            yaml_content = f"invalid-\n{yaml_content}"
        runtime_param_values_stream.write(yaml_content.encode())
        runtime_param_values_stream.flush()

        return runtime_params_filepath

    @pytest.mark.parametrize("use_runtime_params_env_var", [True, False])
    def test_runtime_parameters_invalid_yaml(
        self, resources, tmp_path, runtime_param_values_stream, use_runtime_params_env_var
    ):
        stdout = self._test_custom_model_with_runtime_params(
            resources,
            tmp_path,
            runtime_param_values_stream,
            is_invalid_yaml=True,
            use_runtime_params_env_var=use_runtime_params_env_var,
        )
        assert "Invalid runtime parameter values YAML content!" in stdout

    @pytest.mark.parametrize("use_runtime_params_env_var", [True, False])
    def test_runtime_parameters_missing_attr(
        self, resources, tmp_path, runtime_param_values_stream, use_runtime_params_env_var
    ):
        stdout = self._test_custom_model_with_runtime_params(
            resources,
            tmp_path,
            runtime_param_values_stream,
            is_missing_attr=True,
            use_runtime_params_env_var=use_runtime_params_env_var,
        )
        assert re.search(
            r".*Failed to load runtime parameter.*{\\'credentialType\\': DataError\(\\'is required\\'\)}.*",
            stdout,
        )

    def test_runtime_parameters_boolean_invalid(
        self, resources, tmp_path, runtime_param_values_stream
    ):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, runtime_param_values_stream, bool_var_value="text"
        )
        assert re.search(
            r".*Failed to load runtime parameter.*value should be True or False.*",
            stderr,
        )

    def test_runtime_parameters_numeric_invalid(
        self, resources, tmp_path, runtime_param_values_stream
    ):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, runtime_param_values_stream, numeric_var_value="text"
        )
        assert re.search(
            r".*Failed to load runtime parameter.*value can.*t be converted to float.*",
            stderr,
        )
