"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
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

# Dense preprocessing pipeline, for models such as XGboost that do not do well with
# extremely wide, sparse data
# This preprocessing will work with linear models such as Ridge too
dense_preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessing", sparse_preprocessing_pipeline),
        ("SVD", TruncatedSVD(n_components=10, random_state=42, algorithm="randomized")),
    ]
)


def make_pipeline():
    return sparse_preprocessing_pipeline
    # return dense_preprocessing_pipeline
