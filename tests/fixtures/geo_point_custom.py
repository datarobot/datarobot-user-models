"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import xgboost
import os

def load_model(input_dir):
    model = xgboost.XGBRegressor(objective='reg:squarederror')

    model_path = "xgb_geo_model.json"
    model = model.load_model(os.path.join(input_dir, model_path))
    return model


def score(data, model, **kwargs):
    x = data[['latitude','longitude']]
    x_data = x.values
    predictions = model.predict(x_data)
    return pd.DataFrame(predictions, columns=["longitude", "latitude"])
