# Copyright 2023 DataRobot, Inc. and its affiliates.
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.

"""
This is an example for unstructured custom inference model with MLOps reporting. It is actually
a binary model, which was defined in DataRobot as `Unstructured (Binary)`, just to demonstrate
the usage of the `mlops` instance.
The
"""

import pickle
import time
from pathlib import Path
import sys

import pandas as pd
import tempfile


def load_model(input_dir):
    model_path = str(Path(input_dir) / "model.pkl")
    print(f"Loading model: {model_path}")
    return pickle.load(open(model_path, "rb"))


def score_unstructured(model, data, query, **kwargs):
    print(f"Model: {model} ", flush=True)
    print(f"Incoming data type: {type(data)}", flush=True)
    print(f"Incoming kwargs: {kwargs}", flush=True)
    print(f"Incoming query params: {query}", flush=True)

    mlops = kwargs.get("mlops")

    df = _get_prediction_request_dataframe(data)

    association_ids = df.iloc[:, 0].tolist()
    request_array = df.to_numpy()

    # make predictions
    start_time = time.time()
    predictions_array = model.predict_proba(request_array)
    end_time = time.time()

    if mlops:
        mlops.report_deployment_stats(
            predictions_array.shape[0],  # The number of predictions
            (end_time - start_time) * 1000,  # Prediction execution's time
        )
    else:
        print("Skip mlops reporting because mlops is not enabled.", flush=True)

    reporting_predictions = _prepare_reporting_predictions(predictions_array)

    if mlops:
        mlops.report_predictions_data(
            features_df=df, predictions=reporting_predictions, association_ids=association_ids,
        )
    return str(reporting_predictions)


def _get_prediction_request_dataframe(data):
    with tempfile.NamedTemporaryFile() as f:
        f.write(data.encode("utf-8"))
        f.flush()

        col_names = pd.read_csv(f.name, nrows=0).columns
        types_dict = {"id": str}
        types_dict.update({col: float for col in col_names if col not in types_dict})
        return pd.read_csv(f.name, dtype=types_dict)


def _prepare_reporting_predictions(predictions_array):
    # Based on prediction value and the threshold assign correct label to each prediction
    reporting_predictions = []
    for value in predictions_array.tolist():
        if len(value) == 1:
            # Random forest classifier from scikit-learn can return a single probability value
            # instead of 2 values.  We need to infer the other one before reporting predictions,
            # because, 'report_predictions_data' expects probability for each class.
            value.append(1 - value[0])
        reporting_predictions.append(value)

    return reporting_predictions
