import os

import pytest
import responses

from datarobot_drum.drum.root_predictors.drum_inline_utils import drum_inline_predictor
from datarobot_drum.drum.enum import TargetType
from openai.types.chat.chat_completion import ChatCompletion

from tests.constants import TESTS_FIXTURES_PATH


class TestDrumInlinePredictor:
    @pytest.fixture
    def chat_code_dir(self):
        return os.path.join(TESTS_FIXTURES_PATH, "python3_dummy_chat")

    @pytest.fixture
    def chat_request_no_stream(self):
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke about penguins."},
            ],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 256,
            "n": 1,
            "stream": False,
            "stop": None,
        }

    @pytest.fixture
    def datarobot_details(self):
        return ("http://my-datarobot/api/v2", "notatoken")

    @pytest.skip(
        "Installing datarobot-moderations in functional tests adds 30m to the test suite. Figure out why"
    )
    @responses.activate
    @pytest.mark.parametrize(
        "target_type", [TargetType.AGENTIC_WORKFLOW, TargetType.TEXT_GENERATION]
    )
    def test_chat_with_moderations(
        self,
        target_type,
        monkeypatch,
        chat_code_dir,
        chat_request_no_stream,
        datarobot_details,
    ):
        # arrange
        endpoint, api_token = datarobot_details
        monkeypatch.setitem(os.environ, "DATAROBOT_ENDPOINT", endpoint)
        monkeypatch.setitem(os.environ, "DATAROBOT_API_TOKEN", api_token)

        responses.add(
            responses.GET,
            endpoint + "/version/",
            json={"major": 2, "minor": 37, "versionString": "2.37.0", "releasedVersion": "2.36.0"},
            status=200,
        )

        # act
        with drum_inline_predictor(
            target_type=TargetType.TEXT_GENERATION.value,
            custom_model_dir=chat_code_dir,
            target_name="response",
        ) as predictor:
            result = predictor.chat(chat_request_no_stream)

            # assert
            assert isinstance(result, ChatCompletion)
            assert (
                result.datarobot_moderations["unmoderated_response"]
                == "Echo: You are a helpful assistant."
            )
            assert result.datarobot_moderations["Prompts_token_count"] == 8
            assert result.datarobot_moderations["Responses_token_count"] == 8
            assert result.choices[0].message.content == "Echo: You are a helpful assistant."
            assert result.choices[0].finish_reason == "stop"

    @pytest.mark.parametrize(
        "target_type", [TargetType.AGENTIC_WORKFLOW, TargetType.TEXT_GENERATION]
    )
    def test_chat(self, target_type, monkeypatch, chat_code_dir, chat_request_no_stream):
        # arrange
        monkeypatch.delitem(os.environ, "DATAROBOT_ENDPOINT", raising=False)
        monkeypatch.delitem(os.environ, "DATAROBOT_API_TOKEN", raising=False)

        # act
        with drum_inline_predictor(
            target_type=TargetType.TEXT_GENERATION.value,
            custom_model_dir=chat_code_dir,
            target_name="response",
        ) as predictor:
            result = predictor.chat(chat_request_no_stream)

            # assert
            assert isinstance(result, ChatCompletion)
            assert (
                result.datarobot_moderations["unmoderated_response"]
                == "Echo: You are a helpful assistant."
            )
            assert result.choices[0].message.content == "Echo: You are a helpful assistant."
            assert result.choices[0].finish_reason == "stop"
