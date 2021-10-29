"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd


def load_model(input_dir):
    return "dummy"


def transform(data, model):
    data = data.fillna(0)
    return data


def score(data, model, **kwargs):
    predictions = pd.DataFrame([1.0] * data.shape[0], columns=["Predictions"])
    return predictions
