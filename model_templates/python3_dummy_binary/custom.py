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

    This dummy implementation returns a dataframe with 0.75 probability for the positive class
    and 0.25 for the negative class, regardless of the provided input dataset.

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method
    In case of binary classification model class labels will be provided as the following arguments:
    - `positive_class_label` is the positive class label for a binary classification model
    - `negative_class_label` is the negative class label for a binary classification model

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Binary Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0
    """
    positive_label = kwargs["positive_class_label"]
    negative_label = kwargs["negative_class_label"]
    preds = pd.DataFrame([[0.75, 0.25]] * data.shape[0], columns=[positive_label, negative_label])
    return preds
