"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd


def is_sparse_dataframe(dataframe: pd.DataFrame) -> bool:
    return hasattr(dataframe, "sparse")


def is_sparse_series(series: pd.Series) -> bool:
    return hasattr(series, "sparse")
