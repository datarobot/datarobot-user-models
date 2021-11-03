"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pickle
import string
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=np.object)


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Sparse preprocessing pipeline, for models such as Ridge that handle sparse input well
sparse_preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_selector),
        ("cat", categorical_pipeline, categorical_selector),
    ]
)


def make_pipeline():
    return sparse_preprocessing_pipeline


def random_str_generator(size=4, chars=string.ascii_uppercase):
    return "".join(random.choice(chars) for _ in range(size))


def fit(
    X: pd.DataFrame, y: pd.Series, output_dir: str, **kwargs,
):
    """
    train something to make drum happy
    """
    transformer = make_pipeline()
    transformer.fit(X, y)

    # You must serialize out your transformer to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model or transformer
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)


def transform(X, transformer, y=None):
    """
    just ignore the transformer and return a random df of strings
    """
    return pd.DataFrame([random_str_generator() for x in range(len(X))])
