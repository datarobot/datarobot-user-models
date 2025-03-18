import typing
import pytest
import os
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


@pytest.mark.parametrize("target_type", TargetType.all())
def test_supports_chat(target_type):
    predictor = TestGPUPredictor()
    params = {
        "target_type": target_type,
        "__custom_model_path__": "/opt/code/custom.py",
    }
    os.environ["TARGET_NAME"] = "target_name"
    try:
        predictor.configure(params)
        if target_type == TargetType.TEXT_GENERATION:
            assert predictor.supports_chat()
        else:
            assert not predictor.supports_chat()
    finally:
        os.environ.pop("TARGET_NAME")
