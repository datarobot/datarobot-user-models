# This custom estimator task implements a linear classifier with SGD training

from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier


def fit(
    X: pd.DataFrame,  # input data that DR passes when this task is being trained
    y: pd.Series,  # project's target column
    output_dir: str,  # a path to the output folder; the artifact containing the trained object [in this example - trained SGDClassifier] must be saved into this folder
    class_order: Optional[
        List[str]
    ] = None,  # [for a binary estimator] a list containing names of classes: first, the one that is considered negative inside DR, then the one that is considered positive
    row_weights: Optional[
        np.ndarray
    ] = None,  # a list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings
    **kwargs,
) -> None:  # it doesn't return anything, but must output artifact with the trained object into output_dir

    # This hook defines how DR will train the task
    # It must output the trained object into output_dir

    # fit a SGDClassifier
    estimator = SGDClassifier(loss="log")
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained SGDClassifier]
    # into an artifact [in this example - artifact.pkl]
    # and save it into output_dir so that it can be used later when scoring data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


def score(
    data: pd.DataFrame,  # data that needs to be scored
    model,  # the trained object, extracted from the artifact [in this example - trained SGDClassifier extracted from artifact.pkl]
    **kwargs,
) -> pd.DataFrame:  # returns scored data

    # This hook defines how DR will use the trained object (stored in the variable `model`) to score new data
    # It returns scored data

    # In case of classification, must return a dataframe with a column per class
    # with class names used as column names
    # and probabilities of classes as values (each row must sum to 1.0)
    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
