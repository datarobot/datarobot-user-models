"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd

def load_model(input_dir):
    return "placeholder"


def score(data, model, **kwargs):
    predictions = []
    for _, row in data.iterrows():
        message = row["prompt"]
        # Custom model for testing simply converts the prompt to upper case
        predictions.append(message.upper())
    return pd.DataFrame(predictions, columns=["COMPLETION"])
