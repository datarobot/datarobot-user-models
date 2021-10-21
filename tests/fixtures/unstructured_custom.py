"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""


def load_model(input_dir):
    return "dummy"


def score_unstructured(model, data, query, **kwargs):
    if isinstance(data, bytes):
        data = data.decode("utf8")

    words_count = data.count(" ") + 1

    ret_mode = query.get("ret_mode", "")
    if ret_mode == "binary":
        ret_data = words_count.to_bytes(4, byteorder="big")
        ret_kwargs = {"mimetype": "application/octet-stream"}
        ret = ret_data, ret_kwargs
    else:
        ret = str(words_count)

    return ret
