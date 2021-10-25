"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pickle


def load_model(input_dir):
    """
    Modify this method to deserialize you model if this environment's standard model
    loader cannot. For example, if your custom model archive contains multiple pickle
    files, you must explicitly load which ever one corresponds to your serialized model
    here.

    Parameters
    ----------
    input_dir: string, custom model folder path


    Returns
    -------
    object, the deserialized model
    """
    model_filename = None
    for file in os.listdir(input_dir):
        if file.lower().endswith("sklearn_reg.pkl"):
            model_filename = os.path.join(input_dir, file)
            break

    with open(model_filename, "rb") as picklefile:
        try:
            model = pickle.load(picklefile, encoding="latin1")
        except TypeError:
            model = pickle.load(picklefile)
        return model


def transform(data, model):
    """
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
    for target_col in ["Grade 2014", "Species"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)
    data = data.fillna(0)
    return data
