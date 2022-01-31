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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        tf.random.set_seed(1234)
        input_dim, output_dim = len(X.columns), 1

        model = Sequential(
            [
                Dense(
                    input_dim, activation="relu", input_dim=input_dim, kernel_initializer="normal"
                ),
                Dense(input_dim // 2, activation="relu", kernel_initializer="normal"),
                Dense(output_dim, kernel_initializer="normal"),
            ]
        )
        model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

        callback = EarlyStopping(monitor="loss", patience=3)
        model.fit(
            X, y, epochs=20, batch_size=8, validation_split=0.33, verbose=1, callbacks=[callback]
        )

        # Attach the model to our object for future use
        self.estimator = model
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
        keras.models.save_model(self.estimator, Path(artifact_directory) / "model.h5")
        self.estimator = None

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

        custom_task.estimator = keras.models.load_model(Path(artifact_directory) / "model.h5")

        return custom_task

    def predict(self, X, **kwargs):
        # Note how the regression estimator only outputs one column, so no explicit column names are needed
        return pd.DataFrame(data=self.estimator.predict(X))
