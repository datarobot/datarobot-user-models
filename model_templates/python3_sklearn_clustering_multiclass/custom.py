# Copyright 2023 DataRobot, Inc. and its affiliates.
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.

"""
This is an example for supporting clustering models within DataRobot as custom multiclass models.
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
    """Load model hook for the custom model 

    Parameters
    ----------
    input_dir : str
        the current working directory from which the model artifacts will 

    Returns
    -------
    _type_
        _description_
    """
    model_path = str(Path(input_dir) / "model.pkl")
    class_labels = Path(__file__).parent / Path("class_labels.txt")
    class_labels = class_labels.read_text().split("\n")
    print(f"Loading model: {model_path}")
    return ClusterModel(pickle.load(open(model_path, "rb")), class_labels)



def score(data, model: ClusterModel, **kwargs):
    """Scoring hook for the custom model. 


    Parameters
    ----------
    data : pd.DataFrame
        The input data for the custom model
    model : ClusterModel
        NamedTuple which is the output of the load model hook. 

    Returns
    -------
    pd.DataFrame
        The return data in the form of a pandas dataframe with one column for each output class. 
    """
    
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