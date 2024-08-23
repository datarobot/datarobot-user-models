from typing import Optional
from unittest.mock import patch

import pytest

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionChunk,
    chat_completion_chunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from datarobot_drum.drum.enum import CustomHooks

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter


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
