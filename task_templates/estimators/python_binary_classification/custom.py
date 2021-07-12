# This custom estimator task implements a linear classifier with SGD training

from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier


def fit(X, y, output_dir, class_order, row_weights, **kwargs):
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
    output_dir: str
        A path to the output folder; the artifact [in this example - containing the trained SGDClassifier] must be saved into this folder.
    class_order: list
        [for a binary estimator] a list containing names of classes: first, the one that is considered negative inside DR, then the one that is considered positive.
    row_weights: np.ndarray (optional, default = None)
        A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    """

    # fit SGDClassifier
    estimator = SGDClassifier(loss="log")
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained SGDClassifier]
    # into an artifact [in this example - artifact.pkl]
    # and save it into output_dir so that it can be used later when scoring data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


def score(data, model, **kwargs):
    """ This hook defines how DataRobot will use the trained object from fit() to score new data.
    DataRobot runs this hook when the task is used for scoring inside a blueprint. 
    As an output, this hook is expected to return the scored data.
    The input parameters are passed by DataRobot based on dataset and blueprint configuration.

    Parameters
    -------
    data: pd.DataFrame
        Data that DataRobot passes for scoring.
    model: Any
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained SGDClassifier extracted from artifact.pkl.
    
    Returns
    -------
    pd.DataFrame
        Returns a dataframe with scored data
        In case of classification, must return a dataframe with a column per class
        with class names used as column names
        and probabilities of classes as values (each row must sum to 1.0)
    """

    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
