import pytest
from datarobot_drum.drum.gpu_predictors.vllm_predictor import VllmPredictor
from datarobot_drum.drum.exceptions import UnrecoverableConfigurationError


class TestVllmConfigParsing:
    """
    Unit tests for the engine_config.json parsing logic in the VllmPredictor. We want to ensure that malformed engine_config.json file will raise an exception, exception will be caught and will crash DRUM
    """

    @pytest.fixture
    def predictor(self, mocker):
        """
        Provides a VllmPredictor instance with a patched __init__ and all
        necessary attributes mocked for testing the config parsing logic.
        """
        mocker.patch(
            "datarobot_drum.drum.gpu_predictors.vllm_predictor.VllmPredictor.__init__",
            return_value=None,
        )

        predictor_instance = VllmPredictor()

        # Mock all attributes that are accessed by the method under test
        predictor_instance.logger = mocker.MagicMock()
        predictor_instance._code_dir = "/mock/code/dir"
        predictor_instance.openai_host = "0.0.0.0"
        predictor_instance.openai_port = "9999"
        predictor_instance.served_model_name = "test-model"
        predictor_instance.run_load_model_hook_idempotent = mocker.MagicMock()
        predictor_instance.status_reporter = mocker.MagicMock()
        predictor_instance.openai_process = mocker.MagicMock()
        predictor_instance.model = None
        predictor_instance.huggingface_token = None
        predictor_instance.max_model_len = None
        predictor_instance.gpu_memory_utilization = None
        predictor_instance.gpu_count = 0

        return predictor_instance

    def test_loads_valid_config_successfully(self, mocker, predictor):
        """
        Verifies that when engine_config.json is well-formed, the method
        executes without raising an exception.
        """
        # ARRANGE: Provide valid, well-formed JSON content.
        valid_json = '{"args": ["--model", "meta-llama/Llama-3.1-8B-Instruct"]}'

        vllm_module_path = "datarobot_drum.drum.gpu_predictors.vllm_predictor"
        mocker.patch(f"{vllm_module_path}.Path.is_file", return_value=True)
        mocker.patch(f"{vllm_module_path}.Path.read_text", return_value=valid_json)

        # Also, patch the subprocess call to prevent a real process from starting.
        mock_popen = mocker.patch(f"{vllm_module_path}.subprocess.Popen")

        # ACT & ASSERT:
        try:
            predictor.download_and_serve_model()
        except UnrecoverableConfigurationError as e:
            pytest.fail(f"The method incorrectly crashed on valid JSON: {e}")

        # Verify that the logic proceeded past the config parsing
        mock_popen.assert_called_once()

    @pytest.mark.parametrize(
        "invalid_json",
        [
            '{"args": ["--model", "value",]}',  # Malformed: Trailing comma
            '{"args": ["--model", "value"}',  # Malformed: Missing brace
            '{"args": "not a list"}',  # Invalid Type: String
            '{"args": 12345}',  # Invalid Type: Integer
        ],
    )
    def test_crashes_on_any_invalid_config(self, mocker, predictor, invalid_json):
        """
        Verifies that any malformed or structurally invalid engine_config.json
        correctly raises an UnrecoverableConfigurationError.
        """
        # ARRANGE
        vllm_module_path = "datarobot_drum.drum.gpu_predictors.vllm_predictor"
        mocker.patch(f"{vllm_module_path}.Path.is_file", return_value=True)
        mocker.patch(f"{vllm_module_path}.Path.read_text", return_value=invalid_json)

        # ACT & ASSERT
        with pytest.raises(UnrecoverableConfigurationError):
            predictor.download_and_serve_model()
