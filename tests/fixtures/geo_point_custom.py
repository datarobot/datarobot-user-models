"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd


def load_model(input_dir):
    _ = input_dir
    return "dummy"


def transform(data, model):
    data = data.fillna(0)
    return data


def score(data, model, **kwargs):
    _ = model
    return pd.DataFrame(
        [f"POINT({d.latitude}, {d.longitude})" for _, d in data.iterrows()], columns=["Prediction"]
    )
