import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


def fit(
    X: csr_matrix,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
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


"""
Custom hooks for prediction
---------------------------

If drum's standard assumptions are incorrect for your model, **drum** supports several hooks 
for custom inference code. 
"""
# def init(code_dir : Optional[str], **kwargs) -> None:
#     """
#
#     Parameters
#     ----------
#     code_dir : code folder passed in the `--code_dir` parameter
#     kwargs : future proofing
#     """

# def load_model(code_dir: str) -> Any:
#     """
#     Can be used to load supported models if your model has multiple artifacts, or for loading
#     models that **drum** does not natively support
#
#     Parameters
#     ----------
#     code_dir : is the directory where model artifact and additional code are provided, passed in
#
#     Returns
#     -------
#     If used, this hook must return a non-None value
#     """

# def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
#     """
#     Intended to apply transformations to the prediction data before making predictions. This is
#     most useful if **drum** supports the model's library, but your model requires additional data
#     processing before it can make predictions
#
#     Parameters
#     ----------
#     data : is the dataframe given to **drum** to make predictions on
#     model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
#
#     Returns
#     -------
#     Transformed data
#     """

# def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
#     """
#     This hook is only needed if you would like to use **drum** with a framework not natively
#     supported by the tool.
#
#     Parameters
#     ----------
#     data : is the dataframe to make predictions against. If `transform` is supplied,
#     `data` will be the transformed data.
#     model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
#     kwargs : additional keyword arguments to the method
#     In case of classification model class labels will be provided as the following arguments:
#     - `positive_class_label` is the positive class label for a binary classification model
#     - `negative_class_label` is the negative class label for a binary classification model
#
#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values
#
#     """

# def post_process(predictions: pd.DataFrame, model: Any) -> pd.DataFrame:
#     """
#     This method is only needed if your model's output does not match the above expectations
#
#     Parameters
#     ----------
#     predictions : is the dataframe of predictions produced by **drum** or by
#       the `score` hook, if supplied
#     model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
#
#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row
#     should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values
#
#     """
