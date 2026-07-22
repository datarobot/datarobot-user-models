"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import os
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from datarobot_drum.runtime_parameters.exceptions import RuntimeParameterException
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters

# RAPTOR-18375: strict enforcement is disabled by default. This collision has likely gone
# unnoticed in existing deployments for a long time, so raising unconditionally could break
# custom models that are already (silently) relying on the current behavior.
DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME = (
    "DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS"
)

logger = logging.getLogger(__name__)


def _is_duplicate_extra_model_output_columns_disallowed() -> bool:
    """
    Checked in two places, with the Runtime Parameter taking precedence when both are
    set:
      - MLOPS_RUNTIME_PARAM_<name>: a boolean Runtime Parameter defined on the custom
        model in the Workshop, for a customer to opt in per model.
      - a plain environment variable of the same name, for the platform to inject
        fleet-wide (e.g. once telemetry shows it's safe to enforce by default).
    """
    if RuntimeParameters.has(DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME):
        try:
            return bool(
                RuntimeParameters.get(DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME)
            )
        except RuntimeParameterException:
            logger.warning(
                "Runtime parameter %s is set but could not be read; falling back to the "
                "%s environment variable.",
                DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME,
                DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME,
            )
    return bool(os.environ.get(DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME))


def is_sparse_dataframe(dataframe: pd.DataFrame) -> bool:
    return hasattr(dataframe, "sparse")


def is_sparse_series(series: pd.Series) -> bool:
    return hasattr(series, "sparse")


def check_for_duplicate_extra_model_output_columns(origin_dataframe: pd.DataFrame) -> None:
    duplicated_columns = (
        origin_dataframe.columns[origin_dataframe.columns.duplicated()].unique().tolist()
    )
    if not duplicated_columns:
        return
    message = (
        "Extra model output column name collision. Duplicated column(s): "
        f"{duplicated_columns}. Each additional output column name must be unique, since "
        "'extraModelOutput' returns a single value per name. This is often caused by "
        "re-attaching input data to the score() output under the same name as a computed "
        "output column."
    )
    disallowed = _is_duplicate_extra_model_output_columns_disallowed()
    if disallowed:
        raise ValueError(message)
    logger.warning(
        "%s Only one value per duplicated name will be returned, and which one is undefined. "
        "Set %s=1 to raise instead of warning.",
        message,
        DISALLOW_DUPLICATE_EXTRA_MODEL_OUTPUT_COLUMNS_ENV_VAR_NAME,
    )


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
        check_for_duplicate_extra_model_output_columns(origin_dataframe)
        extra_model_output = origin_dataframe.drop(columns=prediction_columns)
        if len(prediction_columns) == 1:
            ordered_pred_columns = prediction_columns
        else:
            ordered_pred_columns = [
                col for col in original_df_columns_list if col not in extra_model_output.columns
            ]
        predictions = origin_dataframe[ordered_pred_columns]
        return predictions, extra_model_output
