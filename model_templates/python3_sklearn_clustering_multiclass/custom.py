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
from collections import namedtuple

ClusterModel = namedtuple('ClusterModel', ['model', 'class_labels'])

def load_model(input_dir):
    model_path = str(Path(input_dir) / "model.pkl")
    class_labels = Path(__file__).parent / Path("class_labels.txt")
    class_labels = class_labels.read_text().split("\n")
    print(f"Loading model: {model_path}")
    return ClusterModel(pickle.load(open(model_path, "rb")), class_labels)



def score(data, model: ClusterModel, **kwargs):
    
    input_columns = ['Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width']
    for col in input_columns:
        if col not in data.columns.to_list():
            return json.dumps({'message': f"Your data is missing {col} column. Ensure that your data has columns: {input_columns}"})
    clusters = model.model.predict(data[input_columns])
    results= pd.DataFrame({'Prediction': clusters})
    for i in model.class_labels:
        results.loc[:, i] = 0 
        results.loc[results["Prediction"].astype(str) == i, i] = 1
    return results.drop(columns=['Prediction'])