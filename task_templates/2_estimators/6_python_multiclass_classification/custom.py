"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom estimator task implements a decision tree classifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from datarobot_drum.custom_task_interfaces import MulticlassEstimatorInterface


class CustomTask(MulticlassEstimatorInterface):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """ This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column.

        Returns
        -------
        None
            fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
            so that the trained object can be used during scoring.
        """

        # fit DecisionTreeClassifier
        self.estimator = DecisionTreeClassifier()
        self.estimator.fit(X, y)

    def predict_proba(self, data, **kwargs):
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
            In case of classification, must return a dataframe with a column per class
            with class names used as column names
            and probabilities of classes as values (each row must sum to 1.0)
        """

        return pd.DataFrame(
            data=self.estimator.predict_proba(data), columns=self.estimator.classes_
        )
