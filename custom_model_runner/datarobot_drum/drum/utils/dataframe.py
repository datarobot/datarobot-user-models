"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd


def is_sparse_dataframe(dataframe: pd.DataFrame) -> bool:
    return hasattr(dataframe, "sparse")


def is_sparse_series(series: pd.Series) -> bool:
    return hasattr(series, "sparse")


def extract_additional_columns(
    origin_dataframe: pd.DataFrame, prediction_columns: Union[List[str], np.ndarray]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    original_df_columns_list = origin_dataframe.columns.tolist()
    prediction_columns = (
        prediction_columns if isinstance(prediction_columns, list) else list(prediction_columns)
    )
    if prediction_columns == original_df_columns_list:
        return origin_dataframe, None
    else:
        extra_model_output = origin_dataframe.drop(columns=prediction_columns)
        if len(prediction_columns) == 1:
            ordered_pred_columns = prediction_columns
        else:
            ordered_pred_columns = [
                col for col in original_df_columns_list if col not in extra_model_output.columns
            ]
        predictions = origin_dataframe[ordered_pred_columns]
        return predictions, extra_model_output
