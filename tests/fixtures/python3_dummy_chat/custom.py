"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import calendar
import time
from typing import Iterator

from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.model import Model

from datarobot_drum import RuntimeParameters

"""
This example shows how to create a text generation model supporting OpenAI chat
"""

from typing import Any, Dict


def get_supported_llm_models(model: Any):
    """
    Return a list of supported LLM models; response to /v1/models and OpenAI models.list().
    If custom.py does not define this function, DRUM will return a list of either:
    * the model defined in the LLM_ID runtime parameter, if that exists, or:
    * an empty list

    Parameters
    ----------
    model: a model ID to compare against; optional

    Returns: list of openai.types.model.Model
    -------

    """
    return [
        Model(
            id="datarobot_llm_id",
            created=1744854432,
            object="model",
            owned_by="tester@datarobot.com",
        )
    ]


def load_model(code_dir: str) -> Any:
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that **drum** does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """
    return "dummy"


def chat(
    completion_create_params: CompletionCreateParams, model: Any
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """
    This hook supports chat completions; see https://platform.openai.com/docs/api-reference/chat/create.
    In this non-streaming example, the "LLM" echoes back the user's prompt,
    acting as the model specified  in the chat completion request.

    Parameters
    ----------
    completion_create_params: the chat completion request.
    model: the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns: a chat completion.
    -------

    """
    model = completion_create_params["model"]
    message_content = "Echo: " + completion_create_params["messages"][0]["content"]

    return ChatCompletion(
        id="association_id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message_content),
            )
        ],
        created=calendar.timegm(time.gmtime()),
        model=model,
        object="chat.completion",
    )
