"""
    In this example we use an sklearn pipeline that is built using custom classes

    In some cases, avg(prediction) might not match avg(actuals)
    This task, added as a calibrator in the end of a regression blueprint, can help to fix that
    During fit(), it computes and stores the calibration coefficient that is equal to avg(actuals) / avg(predicted) on training data
    During score(), it multiplies incoming data by the calibration coefficient


"""

from typing import List, Optional, Any, Dict
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from custom_pipeline import pipeline  # pipeline defined into custom_pipeline.py


def fit(
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str,
        class_order: Optional[List[str]] = None,
        row_weights: Optional[np.ndarray] = None,
        **kwargs,
) -> None:
    """ This hook MUST ALWAYS be implemented for custom tasks.

    This hook defines how DataRobot will train this task.
    DataRobot runs this hook when the task is being trained inside a blueprint.
    As an output, this hook is expected to create an artifact containg a trained object, that is then used to score new data.
    The input parameters are passed by DataRobot based on project and blueprint configuration.

    Parameters
    -------
    X
        Training data that DataRobot passes when this task is being trained.
    y
        Project's target column.
    output_dir
        A path to the output folder (also provided in --output paramter of 'drum fit' command)
        The artifact [in this example - containing the trained sklearn pipeline]
        must be saved into this folder.
    class_order
        A two element long list dictating the order of classes which should be used for modeling.
        Class order will always be passed to fit by DataRobot for classification tasks,
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
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    """

    # train sklearn pipeline
    estimator = pipeline(X)
    estimator.fit(X, y)

    # dump the trained object [in this example - a trained sklearn pipeline]
    # into an artifact [in this example - artifact.pkl]
    # and save it into output_dir so that it can be used later when scoring data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
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
    data:
        Is the dataframe to make predictions against. If `transform` is supplied,
        `data` will be the transformed data.
    model:
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained sklearn pipeline extracted from artifact.pkl.
    kwargs:
        Additional keyword arguments to the method
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

    return pd.DataFrame(data=model.predict(data), columns=["Predictions"])
