"""
    In this example we show how to create a Pytorch regression or classifiction model
"""
from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from example_code import (
    make_classifier_pipeline,
    make_regressor_pipeline,
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

    This hook MUST ALWAYS be implemented for custom tasks.
    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame
        Training data that DataRobot passes when this task is being trained. Note that both the training data AND
        column (feature) names are passed
    y: pd.Series
        Project's target column. In the case of anomaly detection None is passed
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
    if class_order:
        y = label_binarize(y, classes=class_order)
        estimator = make_classifier_pipeline(X, len(class_order))
    else:
        estimator = make_regressor_pipeline(X)
    estimator.fit(X, y)

    # Dump the trained object [in this example - a trained PyTorch model]
    # into an artifact [in this example - artifact.joblib]
    # and save it into output_dir so that it can be used later when scoring data
    # Note: DRUM will automatically load the model when it is in the default format (see docs)
    # and there is only one artifact file
    serialize_estimator_pipeline(estimator, output_dir)

def load_model(code_dir: str) -> Pipeline:
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
    code_dir: str

    Returns
    -------
    pipelined_model: Pipeline
        Estimator pipeline obj
    """
    return deserialize_estimator_pipeline(code_dir)


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
      Regression: must have a single column called `Predictions` with numerical values
    """

    # Uncomment below for regresssion
    return pd.DataFrame(data=model.predict(data), columns=['Predictions'])

    # Uncomment below for multi-class
    # return pd.DataFrame(data=model.predict_proba(data), columns=kwargs['class_labels'])
