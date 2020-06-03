from __future__ import annotations

from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from model_to_fit import (
    make_classifier_pipeline,
    make_regressor_pipeline,
)


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """
    This hook must be implemented with your fitting code, for running cmrun in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models.
    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X
        training data to perform fit on
    y
        target data to perform fit on
    output_dir
        the path to write output.
        This is the path provided in '--output' parameter of the 'cmrun fit' command.
    class_order
        A two element long list dictating the order of classes which should be used for
        modeling. Class order will always be passed to fit by DataRobot for classification tasks,
        and never otherwise. When models predict, they output a likelihood of one class, with a
        value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
        dictates that the first element in the list will be the 0 class, and the second will be the
        1 class.
    row_weights
        An array of non-negative numeric values which can be used to dictate how important
        a row is. Row weights is only optionally used, and there will be no filtering for which
        custom models support this. There are two situations when values will be passed into
        row_weights, during smart downsampling and when weights are explicitly provided by the user
    kwargs
        Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    # Feel free to delete which ever one of these you aren't using
    if class_order:
        estimator = make_classifier_pipeline(X)
    else:
        estimator = make_regressor_pipeline(X)
    estimator.fit(X, y)

    # NOTE: We currently set a 10GB limit to the size of the serialized model
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


def transform(data, model):
    """
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data : pd.DataFrame
    model : object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if  they're in the dataset
    if "MEDV" in data:
        data.pop("MEDV")
    if "Species" in data:
        data.pop("Species")
    data = data.fillna(0)
    return data
