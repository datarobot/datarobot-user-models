"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom estimator task implements anomaly detection using OneClassSVM

import pandas as pd
from sklearn.svm import OneClassSVM

from datarobot_drum.custom_task_interfaces import AnomalyEstimatorInterface


class CustomTask(AnomalyEstimatorInterface):
    def fit(self, X: pd.DataFrame, y: pd.Series, parameters=None, **kwargs):
        """ This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column. For anomaly, it's None

        Returns
        -------
        None
            fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
            so that the trained object can be used during scoring.
        """

        # fit OneClassSVM
        assert y is None
        self.estimator = OneClassSVM(
            kernel=parameters["kernel"],
            degree=parameters["degree"],
            max_iter=parameters["max_iterations"],
        )
        self.estimator.fit(X, y)

    def predict(self, data, **kwargs):
        """ This hook defines how DataRobot will use the trained object from fit() to score new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the scored data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        data: pd.DataFrame
            Data that DataRobot passes for scoring.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with scored data
            In case of anomaly detection, must return a dataframe with a single column with column name "Predictions"
        """

        return pd.DataFrame(data=self.estimator.predict(data), columns=["Predictions"])
