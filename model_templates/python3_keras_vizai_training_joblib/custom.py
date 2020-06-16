from __future__ import annotations

# pandas/numpy imports
import pandas as pd
import numpy as np

from typing import List, Optional

from model_to_fit import (
    get_transformed_train_test_split,
    fit_image_classifier_pipeline,
    serialize_estimator_pipeline,
    deserialize_estimator_pipeline,
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
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models.
    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame - training data to perform fit on
    y: pd.Series - target data to perform fit on
    output_dir: the path to write output. This is the path provided in '--output' parameter of the
        'drum fit' command.
    class_order : A two element long list dictating the order of classes which should be used for
        modeling. Class order will always be passed to fit by DataRobot for classification tasks,
        and never otherwise. When models predict, they output a likelihood of one class, with a
        value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
        dictates that the first element in the list will be the 0 class, and the second will be the
        1 class.
    row_weights: An array of non-negative numeric values which can be used to dictate how important
        a row is. Row weights is only optionally used, and there will be no filtering for which
        custom models support this. There are two situations when values will be passed into
        row_weights, during smart downsampling and when weights are explicitly provided by the user
    kwargs: Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    # Feel free to delete which ever one of these you aren't using
    if class_order:
        X_train, X_test, y_train, y_test = get_transformed_train_test_split(X, y, class_order)
        fit_estimator = fit_image_classifier_pipeline(X_train, X_test, y_train, y_test)
        # NOTE: We currently set a 10GB limit to the size of the serialized model
        serialize_estimator_pipeline(fit_estimator, output_dir)
    else:
        raise NotImplementedError("Regression not implemented for Visual AI.")
