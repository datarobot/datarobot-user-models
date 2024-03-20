"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import numpy as np
from sklearn.base import BaseEstimator


class MinMaxScaleCapper(object):
    """Min max scaler of a one dimensional array
    This class min-max normalizes the scores returned from
    Anomaly Detection algorithms. Please note sklearn's
    MinMaxScaler does not wotk on 1D arrays
    """

    def __init__(self, t_range=1, y_min=0):
        self.t_range = t_range
        self.y_min = y_min

    def fit(self, y):
        self.y_min = y.min()
        self.t_range = y.max() - self.y_min

        return self

    def transform(self, y_new):
        if self.t_range == 0:
            out = np.ones(len(y_new))
        else:
            out = (y_new - self.y_min) / self.t_range
            out = out.clip(max=1.0, min=0.0)

        return out


# Wrapper to ensure predictions fall between 0 and 1.
# Pass in any anomaly detection model with functions
# 	.fit() and .decision_function()
# For other anomaly models you may not need this wrapper,
# Note that this is a very rudimentary version of
# 	prediction calibration; if you choose to calibrate your own model,
# 	this class may be updated to use any calibration technique with .fit
# 	and .transform methods
class AnomalyCalibEstimator(BaseEstimator):
    """Calibrated anomaly detection estimator"""

    def __init__(self, estimator):
        self.calibrator = MinMaxScaleCapper()
        self.model = estimator

    def fit(self, X, y):
        # Return the classifier
        self.model.fit(X)
        self.calibrator.fit(self.model.decision_function(X).flatten())

        return self

    def predict_proba(self, X):
        scores = self.model.decision_function(X).flatten()

        # Normalize the scores based on the min and max of training data
        # from sklearn docs: "Signed distance is positive for an inlier
        # and negative for an outlier."
        # hence we take 1 - the calibrated scores so 1 corresponds to anomaly
        return 1 - self.calibrator.transform(scores)

    def predict(self, X):
        return self.predict_proba(X)
