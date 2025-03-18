from typing import Optional
from unittest.mock import patch

import pytest

from datarobot_drum.drum.enum import CustomHooks

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter


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
