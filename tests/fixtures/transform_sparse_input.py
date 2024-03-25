"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd
from scipy.sparse import issparse


def transform(X, transformer, y=None):
    """
    Parameters
    ----------
    X: pd.DataFrame - training data to perform transform on
    transformer: object - trained transformer object
    y: pd.Series (optional) - target data to perform transform on
    Returns
    -------
    transformed DataFrame resulting from applying transform to incoming data
    """
    assert all(col.lower().startswith("a") for col in X.columns)
    transformed = transformer.transform(X)
    if issparse(transformed):
        return pd.DataFrame.sparse.from_spmatrix(
            transformed, columns=[f"feature_{i}" for i in range(transformed.shape[1])]
        )
    else:
        return pd.DataFrame(transformed)
