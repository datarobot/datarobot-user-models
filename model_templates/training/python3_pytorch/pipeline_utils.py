#!/usr/bin/env python
# coding: utf-8

# pylint: disable-all
from __future__ import absolute_import

from estimator_utils import PytorchClassifier, PytorchRegressor

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_selector = make_column_selector(
    dtype_include=["int16", "int32", "int64", "float16", "float32", "float64"]
)

num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, numeric_selector)])


def make_regressor(X):
    return Pipeline(
        steps=[("preprocessing", preprocessor), ("model", PytorchRegressor(n_epochs=5))]
    )


def make_classifier(X):
    return Pipeline(
        steps=[("preprocessing", preprocessor), ("model", PytorchClassifier(n_epochs=5))]
    )
