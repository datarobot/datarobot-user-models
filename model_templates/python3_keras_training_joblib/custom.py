from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical

from example_code import (
    make_classifier_pipeline,
    make_regressor_pipeline,
    serialize_estimator_pipeline,
    deserialize_estimator_pipeline,
)

from sklearn.pipeline import Pipeline


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
        y = label_binarize(y, classes=class_order)
        estimator = make_classifier_pipeline(X)
    else:
        estimator = make_regressor_pipeline(X)
    estimator.fit(X, y)

    # NOTE: We currently set a 10GB limit to the size of the serialized model
    serialize_estimator_pipeline(estimator, output_dir)


def load_model(input_dir: str) -> Pipeline:
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    This keras estimator requires 'load_model()' to be overridden. Coz as it involves pipeline of
    preprocessor and estimator bundled together, it requires a special handling (oppose to usually
    simple keras.models.load_model() or unpickling) to load the model. Currently there is no elegant
    default method to save the keras classifier/regressor along with the sklearn pipeline. Hence we
    use deserialize_estimator_pipeline() to load the model pipeline to predict.

    Parameters
    ----------
    input_dir: str

    Returns
    -------
    pipelined_model: Pipeline
        Estimator pipeline obj
    """
    return deserialize_estimator_pipeline(input_dir)
