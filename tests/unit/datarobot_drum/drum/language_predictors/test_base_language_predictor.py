import os
from unittest.mock import patch, Mock, ANY

import pytest
from openai.types.model import Model
import pandas as pd
import numpy as np
import datarobot as dr
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

from custom_model_runner.datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    PythonModelAdapter,
)
from custom_model_runner.datarobot_drum.drum.enum import CustomHooks
from tests.unit.datarobot_drum.drum.chat_utils import create_completion, create_completion_chunks
from tests.unit.datarobot_drum.drum.helpers import MODEL_ID_FROM_RUNTIME_PARAMETER


class TestDRCommonException(Exception):
    pass


class TestLanguagePredictor(BaseLanguagePredictor):
    def supports_chat(self):
        return True

    def _predict(self, **kwargs) -> RawPredictResponse:
        return RawPredictResponse(np.array(["How are you?"]), None)

    def _transform(self, **kwargs):
        pass

    def has_read_input_data_hook(self):
        pass

    def _chat(self, completion_create_params, association_id):
        return self.chat_hook(completion_create_params)


class NoChatLanguagePredictor(BaseLanguagePredictor):
    def _predict(self, **kwargs) -> RawPredictResponse:
        pass

    def _chat(self, completion_create_params, association_id):
        pass

    def _transform(self, **kwargs):
        pass

    def has_read_input_data_hook(self):
        pass


class TestBaseLanguagePredictor:
    def test_base_no_chat(self):
        predictor = NoChatLanguagePredictor()

        assert predictor.supports_chat() == False

    @pytest.fixture
    def language_predictor(self, chat_python_model_adapter):
        predictor = TestLanguagePredictor()
        predictor.configure(
            {
                "target_type": TargetType.TEXT_GENERATION,
                "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
                "external_webserver_url": "http://some.domain",
                "api_token": "1234qwer",
            }
        )

        yield predictor

    @pytest.fixture
    def mock_mlops(self):
        with (
            patch(
                "datarobot_drum.drum.language_predictors.base_language_predictor.mlops_loaded", True
            ),
            patch("datarobot_drum.drum.language_predictors.base_language_predictor.MLOps") as mock,
        ):
            mlops_instance = Mock()
            mock.return_value = mlops_instance
            yield mlops_instance

    @pytest.fixture
    def mock_default_deployment_settings(self):
        with patch.object(dr.Deployment, "get") as mock_get_deployment:
            mock_get_deployment.return_value = Mock()
            mock_get_deployment.return_value.get_champion_model_package.return_value = Mock()
            mock_get_deployment.return_value.model = {"prompt": "promptText"}
            yield

    @pytest.fixture
    def language_predictor_with_mlops(
        self, chat_python_model_adapter, mock_mlops, mock_default_deployment_settings
    ):
        predictor = TestLanguagePredictor()
        predictor.configure(self._language_predictor_with_mlops_parameters())
        yield predictor

    @staticmethod
    def _language_predictor_with_mlops_parameters():
        return {
            "target_type": TargetType.TEXT_GENERATION,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
            "deployment_id": "1234",
            "external_webserver_url": "http://webserver",
            "api_token": "1234qwer",
        }

    def _language_predictor_with_mlops_params_dr_api_access(self):
        params = self._language_predictor_with_mlops_parameters()
        params["allow_dr_api_access"] = True
        return params

    @pytest.fixture
    def mock_dr_client(self):
        with patch.object(dr, "Client") as _:
            yield

    def test_mlops_init(self, language_predictor_with_mlops, mock_mlops):
        # mock_mlops.set_channel_config.assert_called_once_with("spooler_type=API")

        mock_mlops.init.assert_called_once()


class TestChat(TestBaseLanguagePredictor):
    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_without_mlops(self, language_predictor, stream):
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
            messages = []
            for chunk in response:
                assert hasattr(chunk, "datarobot_association_id")
                if chunk.choices[0].delta.content:
                    messages.append(chunk.choices[0].delta.content)
            content = "".join(messages)
        else:
            content = response.choices[0].message.content
            assert hasattr(response, "datarobot_association_id")

        assert content == "How are you"

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

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_with_mlops(self, language_predictor_with_mlops, mock_mlops, stream):
        def chat_hook(completion_request):
            return (
                create_completion_chunks(["How", " are", " you"])
                if stream
                else create_completion("How are you")
            )

        language_predictor_with_mlops.chat_hook = chat_hook

        response = language_predictor_with_mlops.chat(
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
            for chunk in response:
                assert hasattr(chunk, "datarobot_association_id")
        else:
            hasattr(response, "datarobot_association_id")

        mock_mlops.report_deployment_stats.assert_called_once_with(
            num_predictions=1, execution_time_ms=ANY
        )

        mock_mlops.report_predictions_data.assert_called_once_with(
            ANY,
            ["How are you"],
            association_ids=ANY,
        )
        # Compare features dataframe separately as this doesn't play nice with assert_called
        assert (
            mock_mlops.report_predictions_data.call_args.args[0]["promptText"].values[0] == "Hello!"
        )

    def test_missing_required_mlops_parameters(self, chat_python_model_adapter, mock_mlops):
        predictor = TestLanguagePredictor()
        mlops_params = self._language_predictor_with_mlops_parameters()
        for missing_param in ["external_webserver_url", "api_token"]:
            params = mlops_params.copy()
            params.pop(missing_param)
            with pytest.raises(
                ValueError, match=f"MLOps monitoring requires '{missing_param}' parameter"
            ):
                predictor.configure(params)

    def test_association_id(self, language_predictor_with_mlops, mock_mlops):
        with patch.object(TestLanguagePredictor, "_chat") as mock_chat:
            mock_chat.return_value = create_completion("How are you")

            completion = language_predictor_with_mlops.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            )

            association_id = mock_mlops.report_predictions_data.call_args.kwargs["association_ids"][
                0
            ]

            mock_chat.assert_called_once_with(ANY, association_id)
            hasattr(completion, "datarobot_association_id")

    def test_prompt_column_name(self, chat_python_model_adapter, mock_mlops, mock_dr_client):
        language_predictor = TestLanguagePredictor()
        language_predictor_with_mlops_params = (
            self._language_predictor_with_mlops_params_dr_api_access()
        )
        with patch("datarobot.Deployment") as mock_deployment:
            deployment_instance = Mock()
            deployment_instance.model = {"prompt": "newPromptName"}
            deployment_instance.return_value.get_champion_model_package.return_value = Mock()
            mock_deployment.get.return_value = deployment_instance

            language_predictor.configure(language_predictor_with_mlops_params)

        def chat_hook(completion_request):
            return create_completion("How are you")

        language_predictor.chat_hook = chat_hook
        _ = language_predictor.chat(
            {
                "model": "any",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            }
        )

        mock_mlops.report_predictions_data.assert_called_once_with(
            ANY,
            ["How are you"],
            association_ids=ANY,
        )
        # Compare features dataframe separately as this doesn't play nice with assert_called
        assert (
            mock_mlops.report_predictions_data.call_args.args[0]["newPromptName"].values[0]
            == "Hello!"
        )

    @pytest.mark.parametrize("stream", [False, True])
    def test_failing_hook_with_mlops(self, language_predictor_with_mlops, mock_mlops, stream):
        def chat_hook(completion_request):
            raise BadRequest("Error")

        language_predictor_with_mlops.chat_hook = chat_hook

        with pytest.raises(BadRequest):
            language_predictor_with_mlops.chat(
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

    def test_failing_in_middle_of_stream(self, language_predictor_with_mlops, mock_mlops):
        def chat_hook(completion_request):
            def generator():
                for chunk in create_completion_chunks(["Chunk1", "Chunk2"]):
                    yield chunk

                raise BadRequest("Error")

            return generator()

        language_predictor_with_mlops.chat_hook = chat_hook

        with pytest.raises(BadRequest):
            response = language_predictor_with_mlops.chat(
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

    @pytest.mark.parametrize(
        "mlops_function", ["report_predictions_data", "report_deployment_stats"]
    )
    def test_continue_on_mlops_failure(
        self, language_predictor_with_mlops, mock_mlops, mlops_function
    ):
        def chat_hook(completion_request):
            return create_completion("How are you")

        language_predictor_with_mlops.chat_hook = chat_hook

        with patch(
            "datarobot_drum.drum.language_predictors.base_language_predictor.DRCommonException",
            create=True,
            new=TestDRCommonException,
        ):
            mock_mlops_function = getattr(mock_mlops, mlops_function)
            mock_mlops_function.side_effect = TestDRCommonException("fail")

            response = language_predictor_with_mlops.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": False,
                }
            )

        assert response.choices[0].message.content == "How are you"
        mock_mlops_function.assert_called_once()

    @pytest.mark.parametrize(
        "chat_content, expected_prompt",
        [
            ("Hi, How are you?", "Hi, How are you?"),
            (
                [
                    {"type": "text", "text": "Is this image toxic? Answer YES or NO"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://img.url/boardwalk.jpg",
                        },
                    },
                ],
                "Is this image toxic? Answer YES or NO\nImage URL: https://img.url/boardwalk.jpg",
            ),
            (
                [
                    {"type": "text", "text": "Is this audio toxic? Answer YES or NO"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b"encoded string", "format": "wav"},
                    },
                ],
                "Is this audio toxic? Answer YES or NO\nAudio Input, Format: wav",
            ),
        ],
    )
    def test_chat_prompt_inputs(
        self, language_predictor_with_mlops, mock_mlops, chat_content, expected_prompt
    ):
        def chat_hook(completion_request):
            return create_completion("How are you")

        language_predictor_with_mlops.chat_hook = chat_hook
        _ = language_predictor_with_mlops.chat(
            {
                "model": "any",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": chat_content},
                ],
            }
        )
        expected_df = pd.DataFrame({"promptText": [expected_prompt]})
        expected_predictions = ["How are you"]
        actual_df = mock_mlops.report_predictions_data.call_args.args[0]
        actual_predictions = mock_mlops.report_predictions_data.call_args.args[1]
        assert expected_predictions == actual_predictions
        pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True, check_dtype=False)


class TestModelsAPI(TestBaseLanguagePredictor):
    """
    Test serving of models.list() API from a textgen model.
    Where possible, the real adapter is used with no mocking (but minimal configuration).
    """

    def test_wrong_target_type(self):
        """
        Not a textgen model? models() should return empty list
        Prediction server returns HTTP 404;
        that's tested in test_prediction_server_list_llm_models_unsupported.
        """
        os.environ["TARGET_NAME"] = "completion"  # required but not used here
        pma = PythonModelAdapter(model_dir=".", target_type=TargetType.REGRESSION)
        response = pma.get_supported_llm_models(None)
        assert response == {"data": [], "object": "list"}

    def test_no_hook_no_parameter(self):
        """
        models hook is not defined, and LLM_ID runtime parameter does not exist:
        response should succeed but be empty
        """
        os.environ["TARGET_NAME"] = "completion"  # required but not used here
        pma = PythonModelAdapter(model_dir=".", target_type=TargetType.TEXT_GENERATION)
        response = pma.get_supported_llm_models(None)
        assert response == {"data": [], "object": "list"}

    def test_no_hook_with_parameter(self, llm_id_parameter):
        """
        models hook is not defined, but LLM_ID runtime parameter does exist:
        response should use runtime parameter
        """
        os.environ["TARGET_NAME"] = "completion"  # required but not used here
        model_id = MODEL_ID_FROM_RUNTIME_PARAMETER
        pma = PythonModelAdapter(model_dir=".", target_type=TargetType.TEXT_GENERATION)
        response = pma.get_supported_llm_models(None)
        assert response == {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": ANY, "owned_by": "DataRobot"}],
        }

    def test_with_hook(self, llm_id_parameter):
        """
        If models hook is defined: return that hook's results
        Exercises the model adapter's logic; more than just calling our mocked function directly.
        """
        model_id = "test_with_hook model"
        owned_by = "test_with_hook owner"

        def models_hook(model):
            return [
                Model(
                    id=model_id,
                    created=1744854432,
                    object="model",
                    owned_by=owned_by,
                )
            ]

        # provide minimal information to initialize a real adapter
        os.environ["TARGET_NAME"] = "completion"  # required but not used here
        pma = PythonModelAdapter(model_dir=".", target_type=TargetType.TEXT_GENERATION)
        pma._custom_hooks[CustomHooks.GET_SUPPORTED_LLM_MODELS_LIST] = models_hook
        response = pma.get_supported_llm_models(None)
        assert response == {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": ANY, "owned_by": owned_by}],
        }
