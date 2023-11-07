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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

##############################
### Preprocessing tools
##############################

# This selector tells sklearn which columns in a pd.DataFrame are numeric
numeric_selector = make_column_selector(dtype_include=np.number)

# This selector tells sklearn which columns in a pd.DataFrame are categorical
# Note that it will return True for text columns as well
# This means that text variables will be be treated as both text AND categoricals
# This is ok, but if you don't like it, you could write a more complicated is_categorical function
categorical_selector = make_column_selector(dtype_include=object)


def to_string(x):
    """Handle boolean values as string.  They are treated as an object otherwise, and will not work with categorical
    when no missing values are present.  If there are missing values they are already correctly treated as a string.
    """
    return x.astype(str)


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
# 1. Convert boolean values to strings
# 2. Impute missing values with the string "missing"
# 3. One hot encode the data (ignoring new categorical levels at prediction time)
# You can set `handle_unknown='error'` to make your model raise an error at prediction time if
# it encounters a new categorical level
categorical_pipeline = Pipeline(
    steps=[
        ("bool_to_string", FunctionTransformer(to_string)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# Sparse preprocessing pipeline, for models such as Ridge that handle sparse input well
sparse_preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_selector),
        ("cat", categorical_pipeline, categorical_selector),
        # Text preprocessing pipeline has been removed: https://github.com/datarobot/datarobot-user-models/pull/663
        # Either TfidfVectorizer (scikit-learn) or MultiColumnTfidfVectorizer (sagemaker-scikit-learn-extension)
        # can be used to process text.
        # TfidfVectorizer can only handle one column of text at a time,
        # and will fail on datasets with more than one text column.
        # MultiColumnTfidfVectorizer may be useful to handle multiple text columns,
        # but currently it is not compatible with scikit-learn:
        # https://github.com/aws/sagemaker-scikit-learn-extension/issues/42
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


##############################
### Modeling
##############################


def make_regressor(X):
    return Pipeline(
        steps=[("preprocessing", dense_preprocessing_pipeline), ("model", Ridge())], verbose=True
    )
