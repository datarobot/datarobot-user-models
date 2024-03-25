"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


def pipeline(X):
    """
    Simple 2-step sklearn pipeline containing a transform and an estimator steps implemented using custom classes
    It can be used as a calibration task for a regression: add it in the very end of a blueprint,  and it will
    multiply predictions by a fixed coefficients so that, on training, avg(predicted) = avg(actuals)
    """
    return Pipeline(steps=[("preprocessing", Calibrator(X)), ("model", EmptyEstimator())])


class Calibrator:
    """
    During fit(), it computes and stores the calibration coefficient that is equal to
    avg(actuals) / avg(predicted) on training data
    During transform(), it multiplies incoming data by the calibration coefficient
    """

    def __init__(self, X):
        self.multiplier = None  # calibration coefficient
        if len(X.columns) != 1:
            raise Exception(
                "As an input, this task must receive a single column containing predictions of a calibrated estimator. Instead, multiple columns have been passed."
            )

    def fit(self, X, y=None, **kwargs):
        self.multiplier = sum(y) / sum(X[X.columns[0]])
        return self

    def transform(self, X):
        return np.array(X[X.columns[0]] * self.multiplier).reshape(-1, 1)


class EmptyEstimator:
    """
    [Empty] estimator:
    - during fit, it does nothing
    - during predict, it passes incoming data as a prediction
    """

    def fit(self, X, y):
        return self

    def predict(self, data: pd.DataFrame):
        return data[:, 0]
