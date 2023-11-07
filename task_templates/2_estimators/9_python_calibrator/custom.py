"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# In some cases, avg(predictions) might not match avg(actuals)
# This task, added as a calibrator in the end of a regression blueprint, can help to fix that
# During fit(), it computes and stores the calibration coefficient that is equal to avg(actuals) / avg(predicted) on
# training data
# During predict(), it multiplies incoming data by the calibration coefficient

import numpy as np
import pandas as pd

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> None:
        self.calibration_coefficient = sum(y) / sum(X[X.columns[0]])

    def predict(
        self,
        data: pd.DataFrame,
        **kwargs,  # data that needs to be scored
    ) -> pd.DataFrame:  # returns scored data
        # This hook defines how DR will use the trained object (stored in the variable `model`) to score new data

        # In case of regression, must return a dataframe with a single column with column name "Predictions"
        return pd.DataFrame(
            data=np.array(data[data.columns[0]] * self.calibration_coefficient).reshape(-1, 1),
            columns=["Predictions"],
        )
