"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from pathlib import Path
from typing import List, Optional, Any, Dict

import keras.models
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
import pickle

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface

from example_code import (
    build_regressor,
)


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        """ This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containing a trained object, that is then used to predict new data.
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
        self.estimator = build_regressor(X)

        tf.random.set_seed(1234)
        self.estimator.fit(X, y)
        return self

    def save(self, artifact_directory):
        """
        Serializes the object and stores it in `artifact_directory`

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to

        Returns
        -------
        self
        """

        self.estimator.model.save(Path(artifact_directory) / "model")
        # Avoid saving the same (potentially large) model twice
        self.estimator.model = None
        with open(Path(artifact_directory) / "artifact.pkl", "wb") as fp:
            pickle.dump(self, fp)

        return self

    @classmethod
    def load(cls, artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`

        Returns
        -------
        cls
            The deserialized object
        """

        with open(Path(artifact_directory) / "artifact.pkl", "rb") as fp:
            custom_task = pickle.load(fp)

        custom_task.estimator.model = keras.models.load_model(Path(artifact_directory) / "model")

        return custom_task

    def predict(self, X, **kwargs):
        """ This hook defines how DataRobot will use the trained object from fit() to transform new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the transformed data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for transformation.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """

        return pd.DataFrame(data=self.estimator.predict(X))
