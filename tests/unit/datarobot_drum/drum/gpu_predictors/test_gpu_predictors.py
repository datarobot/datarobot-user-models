import typing

from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor
from datarobot_drum.resource.drum_server_utils import DrumServerProcess


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


def test_supports_chat():
    predictor = TestGPUPredictor()
    assert predictor.supports_chat()
