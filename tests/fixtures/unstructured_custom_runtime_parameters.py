"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
from pathlib import Path

from datarobot_drum import RuntimeParameters


EXPECTED_RUNTIME_PARAMS_FILE_NAME = "expected_runtime_parameters.json"


def load_model(input_dir):
    expected_runtime_params_filepath = Path(input_dir) / EXPECTED_RUNTIME_PARAMS_FILE_NAME
    with open(expected_runtime_params_filepath, encoding="utf-8") as fd:
        runtime_params = json.load(fd)

    for runtime_key, runtime_env_value in runtime_params.items():
        actual_payload = RuntimeParameters.get(runtime_key)
        assert actual_payload == runtime_env_value["payload"]

    return "dummy"


def score_unstructured(model, data, query, **kwargs):
    return "Ok"
