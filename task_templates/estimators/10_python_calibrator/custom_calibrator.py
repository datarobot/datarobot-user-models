"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np


class CustomCalibrator(object):
    """
    In some cases, avg(prediction) might not match avg(actuals)
    This class, used as a calibrator for a regression problem, can help to fix that
    During fit(), it computes and stores the calibration coefficient that is equal to avg(actuals) / avg(predicted) on
    training data.
    During predict(), it multiplies incoming data by the calibration coefficient
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

    def predict(self, X):
        return np.array(X[X.columns[0]] * self.multiplier).reshape(-1, 1)
