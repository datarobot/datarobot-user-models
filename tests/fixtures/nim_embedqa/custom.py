"""
Copyright 2025 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json


def score_unstructured(model, data, base_url, openai_client, **kwargs):
    payload = json.loads(data)
    # `input_type` is an extension to OpenAI API that NIM uses for certain embedding models
    # One way to pass this is to append to the model name:
    #   https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html#openai-api
    if "input_type" in payload:
        input_type = payload.pop("input_type")
        model += f"-{input_type}"
    payload["model"] = model
    response = openai_client.embeddings.create(**payload)
    return json.dumps(response), {"mimetype": "application/json"}
