"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd


def transform(data, model):
    # Remove target columns if they're in the dataset
    for target_col in ["Grade 2014", "Species", "class"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)
    return data


def post_process(predictions, model):
    return predictions
