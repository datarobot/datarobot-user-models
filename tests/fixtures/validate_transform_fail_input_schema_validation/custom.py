"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
    assert False, "Should not reach here from input schema validation failing"


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    return data
