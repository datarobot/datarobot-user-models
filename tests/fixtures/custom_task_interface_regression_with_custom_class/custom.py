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

import pandas as pd
from CustomCalibrator import CustomCalibrator  # class defined into CustomCalibrator.py

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
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

        self.estimator = CustomCalibrator(X)
        self.estimator.fit(X, y)

        return self

    def predict(self, X, **kwargs):
        """ This hook defines how DataRobot will use the trained object from fit() to score new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the scored data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for scoring.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with scored data.
            In case of regression, score() must return a dataframe with a single column with column name "Predictions".
        """

        return pd.DataFrame(data=self.estimator.predict(X))
