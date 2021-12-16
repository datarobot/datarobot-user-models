"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from src.regression_interface import BinaryClassificationInterface

def fit(self, X, y, row_weights=None, **kwargs):
    """ This hook defines how DataRobot will train this task.
    DataRobot runs this hook when the task is being trained inside a blueprint.
    As an output, this hook is expected to create an artifact containg a trained object, that is then used to score new data.
    The input parameters are passed by DataRobot based on project and blueprint configuration.

    Parameters
    -------
    X: pd.DataFrame
        Training data that DataRobot passes when this task is being trained.
    y: pd.Series
        Project's target column.
    row_weights: np.ndarray (optional, default = None)
        A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

    Returns
    -------
    CustomTask
        returns an object instance of class CustomTask that can be used in chained method calls
    """

    # fit DecisionTreeClassifier
    estimator = DecisionTreeClassifier()
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained DecisionTreeClassifier]
    # into an artifact [in this example - artifact.pkl]
    # and save it into output_dir so that it can be used later when scoring data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


def score(self, data, **kwargs):
    """ This hook defines how DataRobot will use the trained object from fit() to score new data.
    DataRobot runs this hook when the task is used for scoring inside a blueprint.
    As an output, this hook is expected to return the scored data.
    The input parameters are passed by DataRobot based on dataset and blueprint configuration.

    Parameters
    -------
    data: pd.DataFrame
        Data that DataRobot passes for scoring.

    Returns
    -------
    pd.DataFrame
        Returns a dataframe with scored data.
        In case of regression, score() must return a dataframe with a single column with column name "Predictions".
    """

    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
