"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pickle
import pandas as pd

from datarobot_drum.custom_task_interfaces import TransformerInterface
from scipy.sparse.csr import csr_matrix

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=object)


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


class CustomTask(TransformerInterface):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> None:
        """
        This hook defines how DataRobot will train this task. Even transform tasks need to be trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        ----------
        X: pd.DataFrame - training data to perform fit on
        y: pd.Series - target data to perform fit on
        kwargs: Added for forwards compatibility

        Returns
        -------
        None
        """
        self.transformer = make_pipeline()
        self.transformer.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X: pd.DataFrame - training data to perform transform on

        Returns
        -------
        transformed DataFrame resulting from applying transform to incoming data
        """
        transformed = self.transformer.transform(X)
        if type(transformed) == csr_matrix:
            return pd.DataFrame.sparse.from_spmatrix(transformed)
        else:
            return pd.DataFrame(transformed)
