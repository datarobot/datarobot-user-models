"""
    In this example we show how flexible pipelines can be by creating
    a computer vision pipeline and classifier
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from model_utils import (
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
        value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
        dictates that the first element in the list will be the 0 class, and the second will be the
        1 class.
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
        fit_estimator = fit_image_classifier_pipeline(X, y, class_order)

        # Dump the trained object [in this example - a trained PyTorch model]
        # into an artifact [in this example - artifact.joblib]
        # and save it into output_dir so that it can be used later when scoring data
        # Note: DRUM will automatically load the model when it is in the default format (see docs)
        # and there is only one artifact file
        serialize_estimator_pipeline(fit_estimator, output_dir)
    else:
        raise NotImplementedError("Regression not implemented for Visual AI.")


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
