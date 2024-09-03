import os
from unittest.mock import Mock, patch

import httpx
import openai
import pytest
from httpx import WSGITransport
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
)
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.enum import RunLanguage, TargetType
from datarobot_drum.drum.server import _create_flask_app
from datarobot_drum.resource.components.Python.prediction_server.prediction_server import (
    PredictionServer,
)
from tests.unit.datarobot_drum.drum.chat_utils import create_completion, create_completion_chunks


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
def prediction_server(test_flask_app, chat_python_model_adapter):
    with patch.dict(os.environ, {"TARGET_NAME": "target"}), patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonPredictor._init_mlops"
    ):
        server = PredictionServer(Mock())

        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.TEXT_GENERATION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }

        server.configure(params)
        server._predictor._mlops = Mock()
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
    def chat_hook(completion_request, model):
        assert model == "model"

        assert completion_request["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

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

    def chat_hook(completion_request, model):
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
    def chat_hook(completion_request, model):
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


@pytest.mark.parametrize("processes_param, expected_processes", [(None, 1), (10, 10)])
def test_run_flask_app(processes_param, expected_processes):
    server = PredictionServer(Mock())
    server._params = {"host": "localhost", "port": "6789"}
    if processes_param:
        server._params["processes"] = processes_param

    app = Mock()
    server._run_flask_app(app)
    app.run.assert_called_with("localhost", "6789", threaded=False, processes=expected_processes)
