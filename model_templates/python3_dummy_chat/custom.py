"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import calendar
import time
from typing import Any, Iterator
import uuid

from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.model import Model

# This example shows how to create a text generation model supporting OpenAI chat


def get_supported_llm_models(model: Any):
    """
    Return a list of supported LLM models; response to /v1/models and OpenAI models.list().
    If custom.py does not define this function, DRUM will return a list of either:
    * the model defined in the LLM_ID runtime parameter, if that exists, or:
    * an empty list

    Parameters
    ----------
    model: a model ID to compare against; optional

    Returns
    -------
    List of openai.types.model.Model
    """
    _ = model
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
    _ = code_dir
    return "dummy"


def chat(
    completion_create_params: CompletionCreateParams, model: Any
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """
    This hook supports chat completions;
    see https://platform.openai.com/docs/api-reference/chat/create.
    In this example, the "LLM" echoes back the user's prompt,
    acting as the model specified in the chat completion request.
    If streaming is requested, yields ChatCompletionChunk objects
    for each "token" (word) in the response.
    """
    _ = model
    inter_token_latency_seconds = 0.25
    model_id = completion_create_params["model"]
    message_content = "Echo: " + completion_create_params["messages"][0]["content"]
    stream = completion_create_params.get("stream", False)

    if stream:
        # Mimic OpenAI streaming: yield one chunk at a time, split by whitespace
        def gen_chunks():
            chunk_id = str(uuid.uuid4())
            for token in message_content.split():
                yield ChatCompletionChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=calendar.timegm(time.gmtime()),
                    model=model_id,
                    choices=[
                        ChunkChoice(
                            finish_reason=None,
                            index=0,
                            delta=ChoiceDelta(content=token),
                        )
                    ],
                )
                time.sleep(inter_token_latency_seconds)
            # Send a final chunk with finish_reason
            yield ChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=calendar.timegm(time.gmtime()),
                model=model_id,
                choices=[
                    ChunkChoice(
                        finish_reason="stop",
                        index=0,
                        delta=ChoiceDelta(),
                    )
                ],
            )

        return gen_chunks()
    # non-streaming
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
        model=model_id,
        object="chat.completion",
    )
