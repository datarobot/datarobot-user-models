import pickle
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom tasks.
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
    estimator = Ridge()
    estimator.fit(X, y)

    # You must serialize out your model to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(estimator, fp)

def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook defines the output of a custom estimator and returns predictions on input data.
    It should be skipped if a task is a transform.

    Note: While best practice is to include the score hook, if the score hook is not present DataRobot will
    add a score hook and call the default predict method for the library, e.g. model.predict(X,y) for python

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by DRUM or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method
    In case of classification model class labels will be provided as the following arguments:
    - `positive_class_label` is the positive class label for a binary classification model
    - `negative_class_label` is the negative class label for a binary classification model

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Binary Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0
      Regression: must have a single column called `Predictions` with numerical values
    """

    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)

"""

Additional hooks that are available to use
---------------------------

These hooks are largely meant to help with edge cases around loading libraries / non-standard model artifacts.

"""
# def init(code_dir : Optional[str], **kwargs) -> None:
#     """
#     Can typically be skipped for python, but is required when using R.
#     Allows to load libraries and additional files to use in other hooks.
#
#     Parameters
#     ----------
#     code_dir : code folder passed in the `--code_dir` parameter
#     kwargs : future proofing
#     """

# def load_model(code_dir: str) -> Any:
#     """
#     This hook loads a trained object(s) from the artifact(s).
#     It is only required when a trained object is stored in an artifact
#     that uses an unsupported format or when multiple artifacts are used.
#     See documentation at https://github.com/datarobot/datarobot-user-models for default formats
#
#     Parameters
#     ----------
#     code_dir : is the directory where model artifact and additional code are provided, passed in
#
#     Returns
#     -------
#     If used, this hook must return a non-None value
#     """
