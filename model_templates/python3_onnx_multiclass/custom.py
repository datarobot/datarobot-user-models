"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pickle
from typing import List, Optional, Any, Dict
import numpy as np
import pandas as pd
import onnxruntime

preprocessor = None


def load_model(code_dir: str):
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that DRUM  does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """
    global preprocessor
    with open(os.path.join(code_dir, "preprocessor.pkl"), mode="rb") as f:
        preprocessor = pickle.load(f)

    ort_session = onnxruntime.InferenceSession(os.path.join(code_dir, "multiclass_SDSS.onnx"))
    return ort_session


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if DRUM supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to DRUM to make predictions on
    model : is the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns
    -------
    Transformed data
    """
    # Remove target columns if they're in the dataset
    for target_col in ["class"]:
        if target_col in data:
            data.pop(target_col)
    return data


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    DataRobot will run this hook when the task is used for scoring inside a blueprint

    This hook defines the output of a custom estimator and returns predictions on input data.
    It should be skipped if a task is a transform.

    Note: While best practice is to include the score hook, if the score hook is not present DataRobot will
    add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    Parameters
    ----------
    data: pd.DataFrame
        Is the dataframe to make predictions against. If the `transform` hook is utilized,
        `data` will be the transformed data
    model: Any
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained sklearn pipeline extracted from artifact.pkl.
    kwargs:
        Additional keyword arguments to the method

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0. The original class names defined in the project
        must be used as column names. This applies to binary and multi-class classification.
    """

    # Note how we use the preprocessor that's loaded in load_model
    data = preprocessor.transform(data).astype(np.float32)
    input_names = [i.name for i in model.get_inputs()]
    sess_result = model.run(None, {input_names[0]: data})
    predictions = sess_result[0]
    return pd.DataFrame(data=predictions, columns=kwargs["class_labels"])