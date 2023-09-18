"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

"""
This example shows how to create a multiclass neural net with pytorch
"""

from typing import Any, Dict

import pandas as pd


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
