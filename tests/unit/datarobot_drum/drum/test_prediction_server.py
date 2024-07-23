import os
from typing import Optional
from unittest.mock import Mock, patch

import httpx
import openai
import pytest
from httpx import WSGITransport
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionChunk,
    chat_completion_chunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import RunLanguage, TargetType, CustomHooks
from datarobot_drum.drum.server import _create_flask_app
from datarobot_drum.resource.components.Python.prediction_server.prediction_server import (
    PredictionServer,
)


def create_completion(message_content):
    return ChatCompletion(
        id="id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message_content),
            )
        ],
        created=123,
        model="model",
        object="chat.completion",
    )


def create_completion_chunks(messages):
    def create_chunk(content, finish_reason=None, role=None):
        return ChatCompletionChunk(
            id="id",
            choices=[
                chat_completion_chunk.Choice(
                    delta=ChoiceDelta(content=content, role=role),
                    finish_reason=finish_reason,
                    index=0,
                )
            ],
            created=0,
            model="model",
            object="chat.completion.chunk",
        )

    chunks = []
    #  OpenAI returns a chunk with empty string in beginning of stream
    chunks.append(create_chunk("", role="assistant"))

    for message in messages:
        chunks.append(create_chunk(message))

    #  And a None chunk in the end
    chunks.append(create_chunk(None, finish_reason="stop"))
    return chunks


class ChatPythonModelAdapter(PythonModelAdapter):
    chat_hook = None

    def __init__(self, model_dir, target_type):
        super().__init__(model_dir, target_type)

        self._custom_hooks[CustomHooks.CHAT] = self._call_chat_hook

    def load_model_from_artifact(
        self,
        user_secrets_mount_path: Optional[str] = None,
        user_secrets_prefix: Optional[str] = None,
        skip_predictor_lookup=False,
    ):
        return ""

    def _call_chat_hook(self, model, completion_create_params):
        return ChatPythonModelAdapter.chat_hook(model, completion_create_params)


@pytest.fixture
def test_flask_app():
    with patch("datarobot_drum.drum.server._create_flask_app") as mock_create_flask_app, patch(
        "datarobot_drum.resource.components.Python.prediction_server.prediction_server.PredictionServer._run_flask_app"
    ):
        app = _create_flask_app()
        app.config.update(
            {
                "TESTING": True,
            }
        )

        mock_create_flask_app.return_value = app

        yield app


@pytest.fixture
def chat_python_model_adapter():
    with patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonModelAdapter",
        new=ChatPythonModelAdapter,
    ) as adapter:
        yield adapter


@pytest.fixture
def prediction_server(test_flask_app, chat_python_model_adapter):
    with patch.dict(os.environ, {"TARGET_NAME": "target"}):
        server = PredictionServer(Mock())

        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.TEXT_GENERATION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }
        server.configure(params)
        server.materialize(Mock())


@pytest.fixture
def openai_client(test_flask_app):
    return OpenAI(
        base_url="http://localhost:8080",
        api_key="<KEY>",
        http_client=httpx.Client(transport=WSGITransport(app=test_flask_app)),
    )


@pytest.mark.usefixtures("prediction_server")
def test_prediction_server(openai_client, chat_python_model_adapter):
    def chat_hook(model, completion_request):
        return create_completion("Response")

    chat_python_model_adapter.chat_hook = chat_hook

    completion = openai_client.chat.completions.create(
        model="any",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Response"


@pytest.mark.usefixtures("prediction_server")
@pytest.mark.parametrize("use_generator", [True, False])
def test_streaming(openai_client, chat_python_model_adapter, use_generator):
    chunks = create_completion_chunks(["How", "are", "you", "doing"])

    def chat_hook(model, completion_request):
        return (chunk for chunk in chunks) if use_generator else chunks

    chat_python_model_adapter.chat_hook = chat_hook

    completion = openai_client.chat.completions.create(
        model="any",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        stream=True,
    )

    assert isinstance(completion, Stream)
    chunk_messages = [
        chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content
    ]
    assert chunk_messages == ["How", "are", "you", "doing"]


@pytest.mark.usefixtures("prediction_server")
def test_http_exception(openai_client, chat_python_model_adapter):
    def chat_hook(model, completion_request):
        raise BadRequest("Error")

    chat_python_model_adapter.chat_hook = chat_hook

    with pytest.raises(openai.BadRequestError) as exc_info:
        openai_client.chat.completions.create(
            model="any",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        )

    # Response body should be json with error property
    assert exc_info.value.response.json()["error"] == "Error"
