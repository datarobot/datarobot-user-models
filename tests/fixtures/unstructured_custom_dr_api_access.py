"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot.client import get_client


def load_model(input_dir):
    return "dummy"


def score_unstructured(model, data, query, **kwargs):
    if isinstance(data, bytes):
        data = data.decode("utf8")

    try:
        client = get_client()
    except ValueError:
        client = None

    if client:
        desired_num_version_queries = int(data)
        for counter in range(int(desired_num_version_queries)):
            response = get_client().get("version/")
            print(f"{counter}: {response}", flush=True)

    return "Ok"
