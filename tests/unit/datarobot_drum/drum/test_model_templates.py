"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pytest
from openai import Stream
from openai.types.chat import ChatCompletion

# This module tests score and chat hooks from selected model templates;
# not by direct function call, but via testing flask app, prediction server, and model adapter.
# Rename the imported hooks to keep them distinct
from model_templates.python3_dummy_chat.custom import chat as dummy_chat_chat

# The particular model usually doesn't matter for the example hooks
CHAT_COMPLETIONS_MODEL = "datarobot-deployed-llm"

@pytest.mark.usefixtures("prediction_server")
@pytest.mark.parametrize("is_streaming", [True, False])
def test_dummy_chat_chat(openai_client, chat_python_model_adapter, is_streaming):
    """Test the "python3 dummy chat" hook."""
    chat_python_model_adapter.chat_hook = dummy_chat_chat
    prompt = "Tell me a story"

    completion = openai_client.chat.completions.create(
        model=CHAT_COMPLETIONS_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=is_streaming,
    )

    if is_streaming:
        assert isinstance(completion, Stream)
        chunk_messages = [
            chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content
        ]
        expected_messages = ["Echo:"] + prompt.split()
        assert chunk_messages == expected_messages
    else:
        assert isinstance(completion, ChatCompletion)
        assert completion.choices[0].message.content == "Echo: " + prompt
