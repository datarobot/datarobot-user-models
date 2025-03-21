"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import json

from openai import OpenAI

EMBEDDING_STAGE_MAP = {"indexing": "passage", "prompting": "query"}


def score_unstructured(model: str, data: str, base_url: str, openai_client: OpenAI, **kwargs):
    payload = json.loads(data)

    if "input" not in payload:
        raise ValueError("Field `input` is required.")
    texts = payload["input"]

    if "embedding_stage" not in payload:
        raise ValueError("Field `embedding_stage` is required.")
    embedding_stage = payload["embedding_stage"]
    input_type = EMBEDDING_STAGE_MAP[embedding_stage]

    extra_body = {"truncate": "END"}

    # `input_type` is an extension to OpenAI API that NIM uses for certain embedding models
    # One way to pass this is to append to the model name:
    # https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html#openai-api
    extra_body.update({"input_type": input_type})

    import os

    model_name = os.environ.get("NIM_MODEL_NAME", "datarobot-deployed-llm")
    embedding_response = openai_client.embeddings.create(
        input=texts, model=model_name, encoding_format="float", extra_body=extra_body
    )

    result = [_data.embedding for _data in embedding_response.data]
    return json.dumps({"result": result})