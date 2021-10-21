"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""


def load_model(input_dir):
    return "dummy"


def score_unstructured(model, data, query=None, **kwargs):
    ret_kwargs = {}

    if query["ret_one_or_two"] == "one":
        return data
    elif query["ret_one_or_two"] == "one-with-none":
        return data, None
    else:
        if "ret_mimetype" in query:
            ret_kwargs["mimetype"] = query["ret_mimetype"]
        if "ret_charset" in query:
            ret_kwargs["charset"] = query["ret_charset"]
        return data, ret_kwargs
