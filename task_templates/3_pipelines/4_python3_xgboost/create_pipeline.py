"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
"""
    This is a helper file for the custom.py main file
    which actually creates the pipeline, which consists of data preprocessing steps and an XGBoost model

    Note: This script can build either a classification or regression model depending on the target

"""
from xgboost import XGBClassifier, XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd


def create_regression_model() -> XGBRegressor:
    """
    Create a regression model.

    Returns
    -------
    XGBRegressor
        XGBoost regressor model
    """
    xg_reg = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=20,
        n_estimators=50,
        seed=123,
    )
    return xg_reg


def create_binary_classification_model() -> XGBClassifier:
    """
    Create a binary classification model.

    Returns
    -------
    XGBClassifier
        XGBoost classifier model
    """
    xg_clf = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=20,
        n_estimators=50,
        seed=123,
    )
    return xg_clf


def create_multiclass_classification_model(num_labels: int) -> XGBClassifier:
    """
    Create a binary classification model.

    Parameters
    ----------
    num_labels
        number of classification labels

    Returns
    -------
    XGBClassifier
        XGBoost classifier model
    """
    xg_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_labels,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=20,
        n_estimators=50,
        seed=123,
    )
    return xg_clf


def make_classifier_pipeline(X: pd.DataFrame, num_labels: int) -> Pipeline:
    """
    Make the classifier pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X
        X containing all the required features for training
    num_labels
        number of classification labels

    Returns
    -------
    Pipeline
        Classifier pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.dropna(axis=1, how="all").select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = (
        create_binary_classification_model()
        if num_labels == 2
        else create_multiclass_classification_model(num_labels)
    )

    # pipeline with preprocessor and estimator bundled
    classifier_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return classifier_pipeline


def make_regressor_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Make the regressor pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X
        X containing all the required features for training

    Returns
    -------
    Pipeline
        Regressor pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.dropna(axis=1, how="all").select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("standardize", StandardScaler())]
    )
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = create_regression_model()

    # pipeline with preprocessor and estimator bundled
    regressor_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return regressor_pipeline
