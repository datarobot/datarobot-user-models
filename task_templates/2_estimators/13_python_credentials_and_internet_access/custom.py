"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from datarobot_drum.custom_task_interfaces import BinaryEstimatorInterface, ApiTokenSecret
import requests


class CustomTask(BinaryEstimatorInterface):
    def fit(self, X: pd.DataFrame, y: pd.Series, parameters=None, **kwargs) -> None:
        """This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column.
        parameters: dict (optional, default = None)
            A dictionary of hyperparameters defined in the model-metadata.yaml file for the task.
        Returns
        -------
        None
        """

        self.get_extra_column(X)

        # fit DecisionTreeClassifier
        self.estimator = DecisionTreeClassifier(
            criterion=parameters["criterion"], splitter=parameters["splitter"]
        )
        self.estimator.fit(X, y)

    def predict_proba(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """This hook defines how DataRobot will use the trained object from fit() to score new data.
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
        self.get_extra_column(data)

        return pd.DataFrame(
            data=self.estimator.predict_proba(data), columns=self.estimator.classes_
        )

    def get_extra_column(self, data):
        """This is just a quick demo of what you _could_ do"""
        api_token: ApiTokenSecret = self.secrets["CREDENTIAL"]
        headers = {"Authorization": f"Bearer {api_token.api_token}"}
        self.log_message(
            f"using api-token: {api_token}. In actually logs, the secret value will be starred out"
        )
        rows = data.shape[0]
        response = requests.post(
            "https://cool-column-maker.com/", headers=headers, json={"rows": rows}
        )
        extra_column = pd.read_json(response.json()["extraColumn"])
        data["Cool Data"] = extra_column
