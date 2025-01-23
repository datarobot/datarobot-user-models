from unittest.mock import patch, Mock, ANY

import pytest
import pandas as pd
import numpy as np
import datarobot as dr
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from tests.unit.datarobot_drum.drum.chat_utils import create_completion, create_completion_chunks


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
        mock_mlops.set_channel_config.called_once_with("spooler_type=API")

        mock_mlops.init.assert_called_once()


class TestPredict(TestBaseLanguagePredictor):
    @pytest.mark.parametrize("training_data_available", [True, False])
    def test_report_predictions_data_invocation(
        self, mock_mlops, mock_dr_client, training_data_available
    ):
        language_predictor = TestLanguagePredictor()
        language_predictor_with_mlops_params = (
            self._language_predictor_with_mlops_params_dr_api_access()
        )
        language_predictor_with_mlops_params["monitor"] = True
        champion_model_package = Mock()
        setattr(champion_model_package, "datasets", {})
        if training_data_available:
            champion_model_package.datasets.update(
                {"training_data_catalog_id": "6781c879d5494fd56c36760a"}
            )
        with patch.object(dr.Deployment, "get") as mock_get_deployment:
            mock_get_deployment.return_value = Mock()
            mock_get_deployment.return_value.get_champion_model_package.return_value = (
                champion_model_package
            )
            mock_get_deployment.return_value.model = {"prompt": "promptText"}

            language_predictor.configure(language_predictor_with_mlops_params)

        data = bytes(pd.DataFrame({"promptText": ["Hello!"]}).to_csv(index=False), encoding="utf-8")
        _ = language_predictor.predict(binary_data=data)

        if training_data_available:
            expected_df = pd.DataFrame({"promptText": ["Hello!"]})
            expected_predictions = ["How are you?"]
            actual_df = mock_mlops.report_predictions_data.call_args.kwargs["features_df"]
            actual_predictions = mock_mlops.report_predictions_data.call_args.kwargs["predictions"]
            assert expected_predictions == actual_predictions
            pd.testing.assert_frame_equal(
                actual_df, expected_df, check_like=True, check_dtype=False
            )
        else:
            mock_mlops.report_predictions_data.assert_not_called()


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
            content = "".join([(chunk.choices[0].delta.content or "") for chunk in response])
        else:
            content = response.choices[0].message.content

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

            language_predictor_with_mlops.chat(
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

    @pytest.mark.parametrize("training_data_available", [True, False])
    def test_report_predictions_data_invocation(
        self, mock_mlops, mock_dr_client, training_data_available
    ):
        language_predictor = TestLanguagePredictor()
        language_predictor_with_mlops_params = (
            self._language_predictor_with_mlops_params_dr_api_access()
        )
        champion_model_package = Mock()
        setattr(champion_model_package, "datasets", {})
        if training_data_available:
            champion_model_package.datasets.update(
                {"training_data_catalog_id": "6781c879d5494fd56c36760a"}
            )
        with patch.object(dr.Deployment, "get") as mock_get_deployment:
            mock_get_deployment.return_value = Mock()
            mock_get_deployment.return_value.get_champion_model_package.return_value = (
                champion_model_package
            )
            mock_get_deployment.return_value.model = {"prompt": "promptText"}

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

        if training_data_available:
            expected_df = pd.DataFrame({"promptText": ["Hello!"]})
            expected_predictions = ["How are you"]
            actual_df = mock_mlops.report_predictions_data.call_args.args[0]
            actual_predictions = mock_mlops.report_predictions_data.call_args.args[1]
            assert expected_predictions == actual_predictions
            pd.testing.assert_frame_equal(
                actual_df, expected_df, check_like=True, check_dtype=False
            )
        else:
            mock_mlops.report_predictions_data.assert_not_called()
