from typing import List, Optional
import pandas as pd
import numpy as np

from model_utils import (
    build_regressor,
    build_classifier,
    train_regressor,
    train_classifier,
    save_torch_model,
    build_preprocessor,
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

    preprocessor, cols = build_preprocessor(X)
    preprocessor.fit(X)
    X_preprocessed = preprocessor.transform(X)
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=cols)

    # Feel free to delete which ever one of these you aren't using
    if class_order:
        estimator, optimizer, criterion = build_classifier(X_preprocessed)
        train_classifier(X_preprocessed, y, estimator, optimizer, criterion)
        artifact_name = "torch_bin.pth"
    else:
        estimator, optimizer, criterion = build_regressor(X_preprocessed)
        train_regressor(X_preprocessed, y, estimator, optimizer, criterion)
        artifact_name = "torch_reg.pth"

    # NOTE: We currently set a 10GB limit to the size of the serialized model
    save_torch_model(estimator, output_dir, artifact_name)


def transform(data, model):
    """
    Apply the same preprocessing on prediction as on fit

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if  they're in the dataset
    preprocessor, cols = build_preprocessor(data)
    preprocessor.fit(data)
    data_transformed = preprocessor.transform(data)
    return pd.DataFrame(data_transformed, columns=cols)

