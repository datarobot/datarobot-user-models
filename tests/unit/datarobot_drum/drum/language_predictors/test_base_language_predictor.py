from unittest.mock import patch, Mock, ANY

import pytest
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from tests.unit.datarobot_drum.drum.chat_utils import create_completion, create_completion_chunks


class TestLanguagePredictor(BaseLanguagePredictor):
    def _supports_chat(self):
        return True

    def _predict(self, **kwargs) -> RawPredictResponse:
        pass

    def _transform(self, **kwargs):
        pass

    def has_read_input_data_hook(self):
        pass

    def _chat(self, completion_create_params):
        return self.chat_hook(completion_create_params)


class TestChat:
    @pytest.fixture
    def mock_mlops(self):
        with patch("datarobot_drum.drum.language_predictors.base_language_predictor.MLOps") as mock:
            mlops_instance = Mock()
            mock.return_value = mlops_instance
            yield mlops_instance

    @pytest.fixture
    def language_predictor(self, chat_python_model_adapter, mock_mlops):
        predictor = TestLanguagePredictor()
        predictor.mlpiper_configure(
            {
                "target_type": TargetType.TEXT_GENERATION,
                "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
            }
        )

        yield predictor

    @pytest.mark.parametrize("response", ["Wrong response", None])
    @pytest.mark.parametrize("stream", [False, True])
    def test_hook_wrong_response_type(
        self, language_predictor, chat_python_model_adapter, stream, response
    ):
        def chat_hook(completion_request):
            return response

        language_predictor.chat_hook = chat_hook

        with pytest.raises(
            Exception,
            match=r"Expected response to be ChatCompletion or Iterable\[ChatCompletionChunk\].*",
        ):
            response = language_predictor.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": stream,
                }
            )

            # Streaming response needs to be consumed for anything to happen
            if stream:
                [chunk for chunk in response]

    def test_mlops_init(self, language_predictor, mock_mlops):
        mock_mlops.set_channel_config.called_once_with("spooler_type=API")

        mock_mlops.init.assert_called_once()

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_with_mlops(self, language_predictor, mock_mlops, stream):
        def chat_hook(completion_request):
            return (
                create_completion_chunks(["How", " are", " you"])
                if stream
                else create_completion("How are you")
            )

        language_predictor.chat_hook = chat_hook

        response = language_predictor.chat(
            {
                "model": "any",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "stream": stream,
            }
        )
        if stream:
            # Streaming response needs to be consumed for anything to happen
            [chunk for chunk in response]

        mock_mlops.report_deployment_stats.assert_called_once_with(
            num_predictions=1, execution_time_ms=ANY
        )

        mock_mlops.report_predictions_data.assert_called_once_with(
            ANY,
            ["How are you"],
            association_ids=ANY,
        )
        # Compare features dataframe separately as this doesn't play nice with assert_called
        assert mock_mlops.report_predictions_data.call_args.args[0]["promptText"].values[0] == "Hello!"


    def test_prompt_column_name(self, language_predictor, mock_mlops):
        def chat_hook(completion_request):
            return create_completion("How are you")

        language_predictor.chat_hook = chat_hook

        response = language_predictor.chat(
            {
                "model": "any",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ]
            }
        )

        mock_mlops.report_predictions_data.assert_called_once_with(
            ANY,
            ["How are you"],
            association_ids=ANY,
        )
        # Compare features dataframe separately as this doesn't play nice with assert_called
        assert mock_mlops.report_predictions_data.call_args.args[0]["prompt"].values[0] == "Hello!"


    @pytest.mark.parametrize("stream", [False, True])
    def test_failing_hook_with_mlops(self, language_predictor, mock_mlops, stream):
        def chat_hook(completion_request):
            raise BadRequest("Error")

        language_predictor.chat_hook = chat_hook

        with pytest.raises(BadRequest):
            response = language_predictor.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": stream,
                }
            )

        mock_mlops.report_deployment_stats.assert_called_once_with(
            num_predictions=0, execution_time_ms=ANY
        )
        mock_mlops.report_predictions_data.assert_not_called()

    def test_failing_in_middle_of_stream(self, language_predictor, mock_mlops):
        def chat_hook(completion_request):
            def generator():
                for chunk in create_completion_chunks(["Chunk1", "Chunk2"]):
                    yield chunk

                raise BadRequest("Error")

            return generator()

        language_predictor.chat_hook = chat_hook

        with pytest.raises(BadRequest):
            response = language_predictor.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": True,
                }
            )

            # Streaming response needs to be consumed for anything to happen
            [chunk for chunk in response]

        mock_mlops.report_deployment_stats.assert_called_once_with(
            num_predictions=0, execution_time_ms=ANY
        )
        mock_mlops.report_predictions_data.assert_not_called()
