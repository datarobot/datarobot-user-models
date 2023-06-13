"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from anomaly_helpers import AnomalyCalibEstimator


##############################
### Preprocessing tools
##############################

# This selector tells sklearn which columns in a pd.DataFrame are numeric
numeric_selector = make_column_selector(dtype_include=np.number)

# This selector tells sklearn which columns in a pd.DataFrame are categorical
# Note that it will return True for text columns as well
# This is ok, but if you don't like it, you could write a more complicated is_categorical function
categorical_selector = make_column_selector(dtype_include=object)


##############################
### Preprocessing
##############################

# For numerics we:
# 1. Impute missing values with the median, adding a missing value indicator
# 2. Then center and scale the data
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)

# For categoricals, we:
# 1. Impute missing values with the string "missing"
# 2. One hot encode the data (ignoring new categorical levels at prediction time)
# You can set `handle_unknown='error'` to make your model raise an error at prediction time if
# it encounters a new categorical level
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


# Modified TruncatedSVD that doesn't fail if n_components > ncols
class MyTruncatedSVD(TruncatedSVD):
    def fit_transform(self, X, y=None):
        if X.shape[1] <= self.n_components:
            self.n_components = X.shape[1] - 1
        return TruncatedSVD.fit_transform(self, X=X, y=y)


# Dense preprocessing pipeline, for models such as XGboost that do not do well with
# extremely wide, sparse data
# This preprocessing will work with linear models such as Ridge too
dense_preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessing", sparse_preprocessing_pipeline),
        ("SVD", MyTruncatedSVD(n_components=10, random_state=42, algorithm="randomized")),
    ]
)


# anomaly pipeline
def make_anomaly():
    return Pipeline(
        steps=[
            ("preprocessing", dense_preprocessing_pipeline),
            ("model", AnomalyCalibEstimator(estimator=OneClassSVM())),
        ],
        verbose=True,
    )
