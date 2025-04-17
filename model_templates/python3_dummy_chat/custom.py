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

"""
This example shows how to create a text generation model supporting OpenAI chat
"""

from typing import Any, Dict

import pandas as pd

from openai.types.model import Model

def get_supported_llm_models(model: Any):
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


def score(data, model, **kwargs):
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Note: While best practice is to include the score hook, if the score hook is not present
    DataRobot will add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    This dummy implementation reverses all input text and returns.

    Parameters
    ----------
    data : is the dataframe to make predictions against.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method
    Returns
    -------
    This method should return results as a dataframe with the following format:
      Text Generation: must have column with target, containing text data for each input row.
    """
    data = list(data["input"])
    flipped = ["".join(reversed(inp)) for inp in data]
    result = pd.DataFrame({"output": flipped})
    return result


def chat(completion_create_params: CompletionCreateParams, model: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """
    This hook supports chat completions; see https://platform.openai.com/docs/api-reference/chat/create.
    
    Parameters
    ----------
    completion_create_params: the chat completion request.
    model: the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns: a chat completion.
    -------

    """
    print(str(completion_create_params))
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

