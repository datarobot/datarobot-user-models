"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from pathlib import Path

import keras.models
import pandas as pd
import tensorflow as tf
import pickle

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface

from example_code import build_regressor


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        """Note how in this fit we use a helper function in a separate file to build our model"""
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

        # If your estimator is not pickle-able, you can serialize it using its native method,
        # i.e. in this case for keras we use model.save, and then set the estimator to none
        keras.models.save_model(self.estimator.model, Path(artifact_directory) / "model")
        self.estimator.model = None

        # Now that the estimator is none, it won't be pickled with the CustomTask class (i.e. this one)
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

        return pd.DataFrame(data=self.estimator.predict(X))
