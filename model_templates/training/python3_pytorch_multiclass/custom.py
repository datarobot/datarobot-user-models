import os
import pickle
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from preprocessing import dense_preprocessing_pipeline
from model_utils import build_classifier, train_classifier, save_torch_model

preprocessor = None


def load_model(code_dir: str) -> Any:
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that DRUM does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """
    global preprocessor
    with open(os.path.join(code_dir, "preprocessor.pkl"), mode="rb") as f:
        preprocessor = pickle.load(f)

    model = torch.load(os.path.join(code_dir, "artifact.pth"))
    model.eval()
    return model


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
    if preprocessor is None:
        raise ValueError("Preprocessor not loaded")

    return preprocessor.transform(data)


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running DRUM in the fit mode.

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

    print("Fitting Preprocessing pipeline")
    preprocessor = dense_preprocessing_pipeline.fit(X)
    lb = LabelEncoder().fit(y)

    # write out the class labels file
    print("Serializing preprocessor and class labels")
    with open(os.path.join(output_dir, "class_labels.txt"), mode="w") as f:
        f.write("\n".join(str(label) for label in lb.classes_))
    with open(os.path.join(output_dir, "preprocessor.pkl"), mode="wb") as f:
        pickle.dump(preprocessor, f)

    print("Transforming input data")
    X = preprocessor.transform(X)
    y = lb.transform(y)

    estimator, optimizer, criterion = build_classifier(X, len(lb.classes_))
    print("Training classifier")
    train_classifier(X, y, estimator, optimizer, criterion)
    artifact_name = "artifact.pth"
    save_torch_model(estimator, output_dir, artifact_name)
