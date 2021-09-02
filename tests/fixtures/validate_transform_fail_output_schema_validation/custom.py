import pickle
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
    # Do nothing, but have a placeholder artifact
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump({"placeholder": "artifact"}, fp)


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Identity transform. Should fail from model-metadata
    """
    return data
