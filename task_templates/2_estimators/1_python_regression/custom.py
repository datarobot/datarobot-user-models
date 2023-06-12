"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom estimator task implements a decision tree regressor

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X: pd.DataFrame, y: pd.Series, parameters=None, **kwargs) -> None:
        """ This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
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
        None
            fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
            so that the trained object can be used during scoring.
        """

        # fit a DecisionTreeRegressor
        self.estimator = DecisionTreeRegressor(
            criterion=parameters["criterion"], splitter=parameters["splitter"]
        )
        self.estimator.fit(X, y)

    def predict(self, data: pd.DataFrame, **kwargs):
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
            Returns a dataframe with scored data.
            In case of regression, score() must return a dataframe with a single column with column name "Predictions".
        """

        return pd.DataFrame(data=self.estimator.predict(data), columns=["Predictions"])
