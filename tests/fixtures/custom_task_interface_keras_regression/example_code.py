"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline

import pandas as pd


def create_regression_model(num_features: int) -> Sequential:
    """
    Create a regression model.

    Parameters
    ----------
    num_features: int
        Number of features in X to be trained with

    Returns
    -------
    model: Sequential
        Compiled regression model
    """
    input_dim, output_dim = num_features, 1

    # create model
    model = Sequential(
        [
            Dense(input_dim, activation="relu", input_dim=input_dim, kernel_initializer="normal"),
            Dense(input_dim // 2, activation="relu", kernel_initializer="normal"),
            Dense(output_dim, kernel_initializer="normal"),
        ]
    )
    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
    return model


def build_regressor(X: pd.DataFrame):
    """
    Make the regressor pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X: pd.DataFrame
        X containing all the required features for training

    Returns
    -------
    regressor_pipeline: Pipeline
        Regressor pipeline with preprocessor and estimator
    """

    return KerasRegressor(
        build_fn=create_regression_model,
        num_features=len(X.columns),
        epochs=20,
        batch_size=8,
        verbose=1,
        validation_split=0.33,
        callbacks=[EarlyStopping(patience=20)],
    )
