# Copyright 2023 DataRobot, Inc. and its affiliates.
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.

"""
This is an example for unstructured custom inference model with MLOps reporting. It is actually
While it is a clustering model we can use it as multi-class now that the number of clusters is known. 

"""

import pickle
import time
from pathlib import Path
import sys
import json

import pandas as pd
import tempfile

def load_model(input_dir):
    model_path = str(Path(input_dir) / "model.pkl")
    print(f"Loading model: {model_path}")
    return pickle.load(open(model_path, "rb"))



def score_unstructured(model, data,  **kwargs):
    print(kwargs)
    input_columns = ['Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width']
    if not (kwargs.get('mimetype')  == 'application/json'):
        return json.dumps({'message': "This custom model only supports JSON as input."})
    try: 
        if isinstance(data, str):
            data = pd.read_json(data, orient='records')
        else:
            data = pd.read_json(data.decode(), orient='records')
    except Exception as e:
        print(f"well this failed")
        print(e)
    
    for col in input_columns:
        if col not in data.columns.to_list():
            return json.dumps({'message': f"Your data is missing {col} column. Ensure that your data has columns: {input_columns}"})
    start_time = time.time()
    clusters = model.predict(data)  
    end_time = time.time()
    if mlops := kwargs.get("mlops") : 
        mlops.report_deployment_stats(
            clusters.shape[0],  # The number of predictions
            (end_time - start_time) * 1000,  # Prediction execution's time
        )
        mlops.report_predictions_data(
            features_df=data, predictions=clusters.tolist()
        )
    return json.dumps(clusters.tolist())

