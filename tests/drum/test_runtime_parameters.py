import json
import os
import re
from pathlib import Path
from unittest.mock import patch

from datarobot_drum.drum.enum import ArgumentsOptions

from datarobot_drum.resource.utils import _create_custom_model_dir
from datarobot_drum.resource.utils import _exec_shell_cmd
from tests.drum.constants import PYTHON_UNSTRUCTURED_RUNTIME_PARAMS
from tests.drum.constants import UNSTRUCTURED
from tests.fixtures.unstructured_custom_runtime_parameters import EXPECTED_RUNTIME_PARAMS_FILE_NAME


class TestRuntimeParameters:
    def test_runtime_parameters_success(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(resources, tmp_path)
        assert not stderr

    def _test_custom_model_with_runtime_params(
        self, resources, tmp_path, is_invalid_json=False, is_missing_attr=False
    ):
        problem = UNSTRUCTURED
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework=None,
            problem=problem,
            language=PYTHON_UNSTRUCTURED_RUNTIME_PARAMS,
        )

        runtime_params_key_value, runtime_param_filepath = self._setup_runtime_parameters(
            custom_model_dir, is_invalid_json=is_invalid_json, is_missing_attr=is_missing_attr
        )

        cmd = (
            f"{ArgumentsOptions.MAIN_COMMAND} score "
            f"--code-dir {custom_model_dir} "
            f"--input {runtime_param_filepath} "
            f"--output {Path(tmp_path) / 'output'} "
            f"--target-type {resources.target_types(problem)}"
        )

        with patch.dict(os.environ, runtime_params_key_value):
            _, _, stderr = _exec_shell_cmd(cmd, err_msg=None, assert_if_fail=False)
        return stderr

    @staticmethod
    def _setup_runtime_parameters(custom_model_dir, is_invalid_json, is_missing_attr):
        expected_runtime_params = {
            "SOME_STR_KEY": {"type": "string", "payload": "Hello"},
            "SOME_AWS_CRED_KEY": {
                "type": "credential",
                "payload": {
                    "credential_type": "s3",
                    "region": "us-west",
                    "aws_access_key_id": "123",
                    "aws_secret_access_key": "abc",
                    "aws_session_token": "456edf",
                },
            },
        }
        if is_missing_attr:
            expected_runtime_params["SOME_AWS_CRED_KEY"]["payload"].pop("aws_access_key_id")

        expected_runtime_param_filepath = Path(custom_model_dir) / EXPECTED_RUNTIME_PARAMS_FILE_NAME
        with open(expected_runtime_param_filepath, "w", encoding="utf-8") as fd:
            json.dump(expected_runtime_params, fd)

        expected_runtime_params_env_value = {
            f"MLOPS_RUNTIME_PARAM_{k}": json.dumps(v) for k, v in expected_runtime_params.items()
        }
        if is_invalid_json:
            for k in expected_runtime_params_env_value.keys():
                expected_runtime_params_env_value[k] += "-invalid"

        return expected_runtime_params_env_value, expected_runtime_param_filepath

    def test_runtime_parameters_invalid_json(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, is_invalid_json=True
        )
        assert "Invalid runtime parameter json payload." in stderr

    def test_runtime_parameters_missing_attr(self, resources, tmp_path):
        stderr = self._test_custom_model_with_runtime_params(
            resources, tmp_path, is_missing_attr=True
        )
        assert "Invalid runtime parameter! " in stderr
        assert re.search(r".*{'aws_access_key_id': DataError\(is required\)}.*", stderr)
