"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd
from custom_calibrator import CustomCalibrator  # class defined into CustomCalibrator.py

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        """ In some cases, avg(prediction) might not match avg(actuals)
        This task uses a calibrator in a helper class to fix that. During fit(), it computes and stores
        the calibration coefficient that is equal to avg(actuals) / avg(predicted) on training data
        """

        self.estimator = CustomCalibrator(X)
        self.estimator.fit(X, y)

        return self

    def predict(self, X, **kwargs):
        """We use the predict function from the CustomCalibrator class to multiply incoming data by
        the calibration coefficient.
        """
        return pd.DataFrame(data=self.estimator.predict(X))
