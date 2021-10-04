"""
In this example we show a complex pipeline with a binary linear model.
"""
import pickle
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
from create_pipeline import make_classifier


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
    """ This hook MUST ALWAYS be implemented for custom tasks.

    This hook defines how DataRobot will train this task.
    DataRobot runs this hook when the task is being trained inside a blueprint.

    DataRobot will pass the training data, project target, and additional parameters based on the project
    and blueprint configuration as parameters to this function.

    As an output, this hook is expected to create an artifact containing a trained object,
    that is then used to score new data.

    Parameters
    ----------
    X: pd.DataFrame
        Training data that DataRobot passes when this task is being trained. Note that both the training data AND
        column (feature) names are passed
    y: pd.Series
        Project's target column.
    output_dir: str
        A path to the output folder (also provided in --output paramter of 'drum fit' command)
        The artifact [in this example - containing the trained sklearn pipeline]
        must be saved into this folder.
    class_order: Optional[List[str]]
        This indicates which class DataRobot considers positive or negative. E.g. 'yes' is positive, 'no' is negative.
        Class order will always be passed to fit by DataRobot for classification tasks,
        and never otherwise. When models predict, they output a likelihood of one class, with a
        value from 0 to 1. The likelihood of the other class is 1 - this likelihood.
        The first element in the class_order list is the name of the class considered negative inside DR's project,
        and the second is the name of the class that is considered positive
    row_weights: Optional[np.ndarray]
        An array of non-negative numeric values which can be used to dictate how important
        a row is. Row weights is only optionally used, and there will be no filtering for which
        custom models support this. There are two situations when values will be passed into
        row_weights, during smart downsampling and when weights are explicitly specified in the project settings.
    kwargs
        Added for forwards compatibility.

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact
        (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    """
    # Feel free to delete which ever one of these you aren't using
    if class_order is not None:
        if y.dtype == np.dtype("bool"):
            y = y.astype("str")
        estimator = make_classifier(X)
    else:
        raise Exception("Running binary estimator: class_order expected to be not None")
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
    DataRobot will run this hook when the task is used for scoring inside a blueprint

    This hook defines the output of a custom estimator and returns predictions on input data.
    It should be skipped if a task is a transform.

    Note: While best practice is to include the score hook, if the score hook is not present DataRobot will
    add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    Parameters
    ----------
    data: pd.DataFrame
        Is the dataframe to make predictions against. If the `transform` hook is utilized,
        `data` will be the transformed data
    model: Any
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained sklearn pipeline extracted from artifact.pkl.
    kwargs:
        Additional keyword arguments to the method

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0. The original class names defined in the project
        must be used as column names. This applies to binary and multi-class classification.
    """

    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
