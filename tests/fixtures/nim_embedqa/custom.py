"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json

from openai import OpenAI


def score_unstructured(model: str, data: str, base_url: str, openai_client: OpenAI, **kwargs):
    payload = json.loads(data)

    # `input_type` is an extension to OpenAI API that NIM uses for certain embedding models
    # One way to pass this is to append to the model name:
    #   https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html#openai-api
    input_type = payload.pop("input_type", None)

    # The user passed a model param so just defer to them
    if "model" in payload:
        pass

    # No model param but input_type is set so generate the model name
    elif input_type:
        model += f"-{input_type}"
        payload["model"] = model

    # Otherwise fall back to what DRUM thinks the model name should be
    else:
        payload["model"] = model
    response = openai_client.embeddings.create(**payload)
    return response.to_json(), {"mimetype": "application/json"}
