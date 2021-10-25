"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# In some cases, avg(prediction) might not match avg(actuals)
# This task, added as a calibrator in the end of a regression blueprint, can help to fix that
# During fit(), it computes and stores the calibration coefficient that is equal to avg(actuals) / avg(predicted) on
# training data
# During score(), it multiplies incoming data by the calibration coefficient

# In this example we use a custom python class, CustomCalibrator, so that we can store the complex state inside the
# object and then re-use it during scoring

from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from custom_calibrator import CustomCalibrator  # class defined into CustomCalibrator.py


def fit(
    X: pd.DataFrame,  # input data that DR passes when this task is being trained
    y: pd.Series,  # project's target column
    output_dir: str,  # a path to the output folder; the artifact containing the trained object
    # [in this example - CustomCalibrator class object] must be saved into this folder
    class_order: Optional[
        List[str]
    ] = None,  # [for a binary estimator] a list containing names of classes:
    # first, the one that is considered negative inside DR, then the one that is considered positive
    row_weights: Optional[
        np.ndarray
    ] = None,  # a list of weights. DataRobot passes it in case of smart downsampling or when weights column is
    # specified in project settings
    **kwargs,
) -> None:  # it doesn't return anything, but must output artifact with the trained object into output_dir

    # This hook defines how DR will train the task
    # It must output the trained object into output_dir

    # train CustomCalibrator - i.e. create a class object, then compute and store the calibration coefficient into the
    # class
    estimator = CustomCalibrator(X)
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained class object `estimator`]
    # into an artifact [in this example - artifact.pkl]
    # and save it into output_dir so that it can be used later when scoring data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


def score(
    data: pd.DataFrame,  # data that needs to be scored
    model,  # the trained object, extracted from the artifact
    # [in this example - trained CustomCalibrator extracted from artifact.pkl]
    **kwargs,
) -> pd.DataFrame:  # returns scored data

    # This hook defines how DR will use the trained object (stored in the variable `model`) to score new data

    # In case of regression, must return a dataframe with a single column with column name "Predictions"
    return pd.DataFrame(data=model.predict(data), columns=["Predictions"])
