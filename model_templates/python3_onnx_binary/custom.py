"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
"""
In this example we show a complex pipeline with a binary linear model.
"""
import pickle
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd

def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if they're in the dataset
    for target_col in ["Id", "Species"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)
    return data


# def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
#     """
#     DataRobot will run this hook when the task is used for scoring inside a blueprint
#
#     This hook defines the output of a custom estimator and returns predictions on input data.
#     It should be skipped if a task is a transform.
#
#     Note: While best practice is to include the score hook, if the score hook is not present DataRobot will
#     add a score hook and call the default predict method for the library
#     See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details
#
#     Parameters
#     ----------
#     data: pd.DataFrame
#         Is the dataframe to make predictions against. If the `transform` hook is utilized,
#         `data` will be the transformed data
#     model: Any
#         Trained object, extracted by DataRobot from the artifact created in fit().
#         In this example, contains trained sklearn pipeline extracted from artifact.pkl.
#     kwargs:
#         Additional keyword arguments to the method
#
#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row should sum to 1.0. The original class names defined in the project
#         must be used as column names. This applies to binary and multi-class classification.
#     """
#
#     return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
