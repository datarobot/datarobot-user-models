import json
import os
import typing
from unittest.mock import patch

import pytest

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.enum import TargetType

from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor
from datarobot_drum.drum.gpu_predictors.nim_predictor import NIMPredictor
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerProcess
from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes


class TestGPUPredictor(BaseOpenAiGpuPredictor):
    @property
    def num_deployment_stages(self):
        pass

    def health_check(self) -> typing.Tuple[dict, int]:
        pass

    def download_and_serve_model(self, openai_process: DrumServerProcess):
        pass

    def terminate(self):
        pass


@pytest.fixture
def mock_target_name_env_var(monkeypatch):
    monkeypatch.setenv("TARGET_NAME", "target")
    yield
    monkeypatch.delenv("TARGET_NAME")


@pytest.fixture
def mock_openai_host_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_HOST", "mocked.openai.host")
    yield
    monkeypatch.delenv("OPENAI_HOST")


@pytest.fixture
def mock_openai_port_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_PORT", "45678")
    yield
    monkeypatch.delenv("OPENAI_PORT")


@pytest.mark.parametrize("target_type", list(TargetType))
def test_supports_chat(mock_target_name_env_var, target_type):
    predictor = TestGPUPredictor()
    params = {
        "target_type": target_type,
        "__custom_model_path__": "/opt/code/custom.py",
    }
    predictor.configure(params)
    if target_type in [TargetType.TEXT_GENERATION, TargetType.AGENTIC_WORKFLOW]:
        assert predictor.supports_chat()
    else:
        assert not predictor.supports_chat()

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
            assert RuntimeParameters.has(runtime_param_name)
            assert RuntimeParameters.get(runtime_param_name) == payload


def rt_param_name(name):
    return RuntimeParameters.namespaced_param_name(name)


def rt_param_value(type, value):
    return json.dumps(
        {
            "type": type,
            "payload": value,
        }
    )


def rt_param_str_value(value):
    return rt_param_value(RuntimeParameterTypes.STRING.value, value)


class TestNIMPredictor:
    def test_nim_predictor_created_with_default_values(self):
        predictor = NIMPredictor()
        assert predictor.health_route == "/v1/health/ready"
        assert predictor.openai_port == "9999"
        assert predictor.openai_host == "localhost"

    @pytest.mark.usefixtures("mock_openai_host_env_var", "mock_openai_port_env_var")
    def test_nim_predictor_created_with_values_from_env_vars(self):
        predictor = NIMPredictor()
        assert predictor.health_route == "/v1/health/ready"
        assert predictor.openai_port == "45678"
        assert predictor.openai_host == "mocked.openai.host"

    @pytest.mark.usefixtures("mock_openai_host_env_var", "mock_openai_port_env_var")
    def test_nim_predictor_created_with_values_from_runtime_params_and_takes_precedence_over_env_vars(
        self,
    ):
        with patch.dict(
            os.environ,
            {
                rt_param_name("DR_NIM_HEALTH_ROUTE"): rt_param_str_value("/overriden/health/route"),
                rt_param_name("DR_NIM_SERVER_HOST"): rt_param_str_value("mocked.host"),
                rt_param_name("DR_NIM_SERVER_PORT"): rt_param_str_value("mocked.port"),
            },
        ):
            predictor = NIMPredictor()

            assert predictor.health_route == "/overriden/health/route"
            assert predictor.openai_port == "mocked.port"
            assert predictor.openai_host == "mocked.host"
