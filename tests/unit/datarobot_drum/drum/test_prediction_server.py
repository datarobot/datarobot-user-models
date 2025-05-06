import os
import uuid
from unittest.mock import ANY
from unittest.mock import Mock, patch

import httpx
import openai
import pytest
from httpx import WSGITransport
from openai import NotFoundError
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
)
from openai.types.model import Model
from werkzeug.exceptions import BadRequest

from datarobot_drum.drum.enum import RunLanguage, TargetType
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.root_predictors.prediction_server import PredictionServer
from datarobot_drum.drum.server import create_flask_app, HEADER_REQUEST_ID
from datarobot_drum.drum.server import get_flask_app
from tests.unit.datarobot_drum.drum.chat_utils import create_completion, create_completion_chunks
from tests.unit.datarobot_drum.drum.helpers import MODEL_ID_FROM_RUNTIME_PARAMETER


@pytest.fixture
def test_flask_app():
    with patch("datarobot_drum.drum.server.create_flask_app") as mock_create_flask_app, patch(
        "datarobot_drum.drum.root_predictors.prediction_server.PredictionServer._run_flask_app"
    ):
        app = create_flask_app()
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
    ), patch.object(LazyLoadingHandler, "download_lazy_loading_files"):
        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.TEXT_GENERATION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }
        server = PredictionServer(params)
        server._predictor._mlops = Mock()
        server.materialize()


@pytest.fixture
def list_models_prediction_server(test_flask_app, list_models_python_model_adapter):
    with patch.dict(os.environ, {"TARGET_NAME": "target"}), patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonPredictor._init_mlops"
    ), patch.object(LazyLoadingHandler, "download_lazy_loading_files"):
        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.TEXT_GENERATION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }
        server = PredictionServer(params)
        server._predictor._mlops = Mock()
        server.materialize()


@pytest.fixture
def non_chat_prediction_server(test_flask_app, non_chat_python_model_adapter):
    with patch.dict(os.environ, {"TARGET_NAME": "target"}), patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonPredictor._init_mlops"
    ), patch.object(LazyLoadingHandler, "download_lazy_loading_files"):
        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.TEXT_GENERATION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }
        server = PredictionServer(params)
        server._predictor._mlops = Mock()
        server.materialize()


@pytest.fixture
def non_textgen_prediction_server(test_flask_app, non_chat_python_model_adapter):
    with patch.dict(os.environ, {"TARGET_NAME": "target"}), patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonPredictor._init_mlops"
    ), patch.object(LazyLoadingHandler, "download_lazy_loading_files"):
        params = {
            "run_language": RunLanguage.PYTHON,
            "target_type": TargetType.REGRESSION,
            "deployment_config": None,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }
        server = PredictionServer(params)
        server._predictor._mlops = Mock()
        server.materialize()


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


@pytest.mark.usefixtures("non_textgen_prediction_server")
def test_prediction_server_chat_unsupported(openai_client):
    """Attempt to chat with a non-textgen model."""
    with pytest.raises(NotFoundError, match="but chat is not supported"):
        _ = openai_client.chat.completions.create(
            model="any",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        )


@pytest.mark.usefixtures("non_chat_prediction_server")
def test_prediction_server_chat_unimplemented(openai_client):
    """Attempt to chat when a textgen model does not implement chat()."""
    with pytest.raises(NotFoundError, match=r"but chat\(\) is not implemented"):
        _ = openai_client.chat.completions.create(
            model="any",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        )


@pytest.mark.usefixtures("prediction_server")
def test_prediction_server_list_llm_models_no_hook_no_rtp(openai_client, chat_python_model_adapter):
    """Attempt to list supported LLM models where no hook exists and no runtime parameter exists."""
    response = openai_client.models.list()
    assert response.object == "list"
    assert response.data == []


@pytest.mark.usefixtures("prediction_server")
def test_prediction_server_list_llm_models_no_hook_with_rtp(
    openai_client, chat_python_model_adapter, llm_id_parameter
):
    """Attempt to list supported LLM models where no hook exists but a runtime parameter exists."""
    response = openai_client.models.list()
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].to_dict() == {
        "id": MODEL_ID_FROM_RUNTIME_PARAMETER,
        "object": "model",
        "created": ANY,
        "owned_by": "DataRobot",
    }


@pytest.mark.usefixtures("list_models_prediction_server")
def test_prediction_server_list_llm_models_with_hook(
    openai_client, list_models_python_model_adapter, llm_id_parameter
):
    """
    List supported LLM models where a hook exists.
    The runtime parameter should be ignored in favor of the hook response.
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

    list_models_python_model_adapter.models_hook = models_hook
    response = openai_client.models.list()
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].to_dict() == {
        "id": model_id,
        "object": "model",
        "created": ANY,
        "owned_by": owned_by,
    }


@pytest.mark.usefixtures("non_textgen_prediction_server")
def test_prediction_server_list_llm_models_unsupported(openai_client):
    """Attempt to list supported LLM models with a non-textgen model."""
    with pytest.raises(NotFoundError, match="is supported only for TextGen models"):
        _ = openai_client.models.list()


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
    params = {
        "host": "localhost",
        "port": "6789",
        "run_language": "python",
        "target_type": "regression",
        "deployment_config": None,
    }
    if processes_param:
        params["processes"] = processes_param

    with patch.object(PredictionServer, "_setup_predictor"):
        server = PredictionServer(params)

    app = Mock()
    server._run_flask_app(app)
    app.run.assert_called_with("localhost", "6789", threaded=False, processes=expected_processes)


@pytest.mark.usefixtures("prediction_server")
def test_request_id_in_flask_app(test_flask_app):
    prediction_client = test_flask_app.test_client()

    sample_request_id = str(uuid.uuid4())
    # test unique request_id generated for the request
    data = prediction_client.get("/info/")
    assert data.headers.get(HEADER_REQUEST_ID)
    assert data.headers.get(HEADER_REQUEST_ID) != sample_request_id

    # test request with propagated request_id
    data = prediction_client.get("/info/", headers={HEADER_REQUEST_ID: sample_request_id})
    assert data.headers.get(HEADER_REQUEST_ID) == sample_request_id

    # make sure the next request without request_id will have unique identifier by default
    data = prediction_client.get("/info/")
    assert data.headers.get(HEADER_REQUEST_ID)
    assert data.headers.get(HEADER_REQUEST_ID) != sample_request_id
