import pytest
from datarobot_drum.drum.gpu_predictors.vllm_predictor import VllmPredictor


@pytest.fixture
def predictor_for_config_test(mocker):
    """
    Provides a VllmPredictor instance with a patched __init__
    and all necessary attributes mocked for testing download_and_serve_model.
    """
    mocker.patch(
        "datarobot_drum.drum.gpu_predictors.vllm_predictor.VllmPredictor.__init__",
        return_value=None,
    )
    predictor = VllmPredictor()

    predictor.logger = mocker.MagicMock()
    predictor.status_reporter = mocker.MagicMock()
    predictor.python_model_adapter = mocker.MagicMock()
    predictor._code_dir = "/mock/code/dir"
    predictor.openai_host = "0.0.0.0"
    predictor.openai_port = "9999"
    predictor.served_model_name = "test-model"
    predictor.run_load_model_hook_idempotent = mocker.MagicMock()
    predictor.model = None
    predictor.huggingface_token = None
    predictor.max_model_len = None
    predictor.gpu_memory_utilization = None
    predictor.gpu_count = 1

    return predictor
