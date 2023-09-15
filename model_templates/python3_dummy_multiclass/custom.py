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
    return 'dummy'


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Note: While best practice is to include the score hook, if the score hook is not present
    DataRobot will add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    This dummy implementation returns a dataframe with columns, representing all provided classes,
    assigning 0.75 probability to the first class, and the rest of probability to other classes,
    regardless of the provided input dataset.

    Parameters
    ----------
    data: pd.DataFrame
        Is the dataframe to make predictions against. If the `transform` hook is utilized,
        `data` will be the transformed data
    model: Any
        Deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs:
        Additional keyword arguments to the method
        In case of multiclass model class labels will be provided in the `class_labels` argument.

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Multiclass: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0.
        The original class names defined in the project must be used as column names.
    """
    class_labels = kwargs["class_labels"]
    M = len(class_labels)
    data = [[0.75] + (M - 1) * [0.25 / (M - 1)]] * data.shape[0]
    predictions = pd.DataFrame(data=data, columns=class_labels)
    return predictions
