import pickle
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from create_pipeline import make_classifier
from sklearn.preprocessing import label_binarize


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if DRUM supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to DRUM to make predictions on
    model : is the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns
    -------
    Transformed data
    """
    if "class" in data.columns:
        data.pop("class")
    return data


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
    # Feel free to delete which ever one of these you aren't using
    if class_order is not None:
        if y.dtype == np.dtype("bool"):
            y = y.astype("str")
        estimator = make_classifier(X)
    else:
        raise Exception("Running binary estimator_tasks: class_order expected to be not None")
    estimator.fit(X, y)

    # You must serialize out your model to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(estimator, fp)
    with open("{}/class_labels.txt".format(output_dir), "wb") as fp:
        fp.write("\n".join(str(class_) for class_ in estimator.classes_).encode("utf-8"))


"""
Custom hooks for prediction
---------------------------

If drum's standard assumptions are incorrect for your model, DRUM supports several hooks
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
#     models that DRUM does not natively support
#
#     Parameters
#     ----------
#     code_dir : is the directory where model artifact and additional code are provided, passed in
#
#     Returns
#     -------
#     If used, this hook must return a non-None value
#     """

# def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
#     """
#     This hook is only needed if you would like to use DRUM with a framework not natively
#     supported by the tool.
#
#     Parameters
#     ----------
#     data : is the dataframe to make predictions against. If `transform` is supplied,
#     `data` will be the transformed data.
#     model : is the deserialized model loaded by DRUM or by `load_model`, if supplied
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
#     predictions : is the dataframe of predictions produced by DRUM or by
#       the `score` hook, if supplied
#     model : is the deserialized model loaded by DRUM or by `load_model`, if supplied
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
