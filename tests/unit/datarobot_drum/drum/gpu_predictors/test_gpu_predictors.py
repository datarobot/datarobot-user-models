import typing
import pytest
from datarobot_drum.drum.enum import TargetType

from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerProcess


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
