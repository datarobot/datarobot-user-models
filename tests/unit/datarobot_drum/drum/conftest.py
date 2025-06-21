import os
from typing import Optional
from unittest.mock import Mock, patch

import httpx
import pytest
from httpx import WSGITransport
from openai import OpenAI

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import CustomHooks, RunLanguage, TargetType
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.root_predictors.prediction_server import PredictionServer
from datarobot_drum.drum.server import create_flask_app
from tests.unit.datarobot_drum.drum.helpers import MODEL_ID_FROM_RUNTIME_PARAMETER
from tests.unit.datarobot_drum.drum.helpers import inject_runtime_parameter
from tests.unit.datarobot_drum.drum.helpers import unset_runtime_parameter


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
        return "model"

    def _call_chat_hook(self, model, completion_create_params):
        return ChatPythonModelAdapter.chat_hook(model, completion_create_params)


@pytest.fixture
def chat_python_model_adapter():
    with patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonModelAdapter",
        new=ChatPythonModelAdapter,
    ) as adapter:
        yield adapter


class ListModelsPythonModelAdapter(PythonModelAdapter):
    """
    An adapter fixture implementing the GET_SUPPORTED_LLM_MODELS_LIST hook.
    The indirect method of setting a hook invalidates the real adapter's logic
    for determining whether the hook was defined in custom.py.
    To simulate a model that does not define the hook, use ChatPythonModelAdapter.
    """

    chat_hook = None
    models_hook = None

    def __init__(self, model_dir, target_type):
        super().__init__(model_dir, target_type)

        self._custom_hooks[CustomHooks.CHAT] = self._call_chat_hook
        self._custom_hooks[CustomHooks.GET_SUPPORTED_LLM_MODELS_LIST] = self._call_models_hook

    def load_model_from_artifact(
        self,
        user_secrets_mount_path: Optional[str] = None,
        user_secrets_prefix: Optional[str] = None,
        skip_predictor_lookup=False,
    ):
        return "model"

    def _call_chat_hook(self, model, completion_create_params):
        return ListModelsPythonModelAdapter.chat_hook(model, completion_create_params)

    def _call_models_hook(self, model):
        return ListModelsPythonModelAdapter.models_hook(model)


@pytest.fixture
def list_models_python_model_adapter():
    with patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonModelAdapter",
        new=ListModelsPythonModelAdapter,
    ) as adapter:
        yield adapter


class NonChatPythonModelAdapter(PythonModelAdapter):
    """A model adapter whose supports_chat() will return False"""

    def __init__(self, model_dir, target_type):
        super().__init__(model_dir, target_type)

        self._custom_hooks.pop(CustomHooks.CHAT, None)

    def load_model_from_artifact(
        self,
        user_secrets_mount_path: Optional[str] = None,
        user_secrets_prefix: Optional[str] = None,
        skip_predictor_lookup=False,
    ):
        return "model"


@pytest.fixture
def non_chat_python_model_adapter():
    with patch(
        "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.PythonModelAdapter",
        new=NonChatPythonModelAdapter,
    ) as adapter:
        yield adapter


@pytest.fixture
def llm_id_parameter():
    """Run this test with the LLM_ID parameter set (and remove afterwards)"""
    parameter_name = "LLM_ID"
    inject_runtime_parameter(parameter_name, MODEL_ID_FROM_RUNTIME_PARAMETER)
    yield
    unset_runtime_parameter(parameter_name)


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
def openai_client(test_flask_app):
    return OpenAI(
        base_url="http://localhost:8080",
        api_key="<KEY>",
        http_client=httpx.Client(transport=WSGITransport(app=test_flask_app)),
    )


@pytest.fixture
def prediction_server(test_flask_app, chat_python_model_adapter):
    _, _ = test_flask_app, chat_python_model_adapter  # depends on fixture side effects
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
