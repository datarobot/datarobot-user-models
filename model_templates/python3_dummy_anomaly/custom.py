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


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Note: While best practice is to include the score hook, if the score hook is not present
    DataRobot will add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Anomaly Detection: must have a single column called `Predictions` with floating point values,
      denoting probability of a respective row being an anomaly.
    """
    # This will mark every 10th row of the input as 75% probability anomaly.
    # Other rows are 0% anomaly.
    predictions = pd.DataFrame([0.0] * data.shape[0], columns=["Predictions"])
    predictions["Predictions"].iloc[[(i + 1) * 10 - 1 for i in range(data.shape[0] // 10)]] = 0.75
    return predictions
