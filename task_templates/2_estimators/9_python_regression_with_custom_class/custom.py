"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# In some cases, avg(prediction) might not match avg(actuals)
# This task, added as a calibrator in the end of a regression blueprint, can help to fix that.
# During fit(), it computes and stores the calibration coefficient that is equal to avg(actuals) / avg(predicted) on training data.
# During score(), it multiplies incoming data by the calibration coefficient.

# In this example we use a custom python class, CustomCalibrator, so that we can store the complex state inside the object and then re-use it during scoring.

from typing import List, Optional
import pickle
import pandas as pd
from pathlib import Path
from CustomCalibrator import CustomCalibrator  # class defined into CustomCalibrator.py


def fit(X, y, output_dir, row_weights, **kwargs):
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
        A path to the output folder; the artifact [in this example - containing the trained CustomCalibrator class object] must be saved into this folder.
    row_weights: np.ndarray (optional, default = None)
        A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    """

    # train CustomCalibrator - i.e. create a class object, then compute and store the calibration coefficient into the class object.
    estimator = CustomCalibrator(X)
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained object `estimator`]
    # into an artifact [in this example - artifact.pkl]
    # and then save it into output_dir so that it can be used later when scoring data
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
        In this example, contains an object of CustomCalibrator class extracted from artifact.pkl.
    
    Returns
    -------
    pd.DataFrame
        Returns a dataframe with scored data.
        In case of regression, must return a dataframe with a single column with column name "Predictions".
    """

    return pd.DataFrame(data=model.predict(data), columns=["Predictions"])
