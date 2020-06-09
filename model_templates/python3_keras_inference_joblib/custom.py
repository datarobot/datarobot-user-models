import pandas as pd
from pathlib import Path

from model_to_fit import deserialize_estimator_pipeline

from sklearn.pipeline import Pipeline


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

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

    # For Boston Housing dataset
    if "MEDV" in data:
        data.pop("MEDV")

    # For sklearn iris dataset
    if "Species" in data:
        data.pop("Species")

    # for Loan Lending Club dataset
    if "is_bad" in data:
        data.pop("is_bad")

    data = data.fillna(0)
    return data


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
    artifact_path = Path(input_dir) / "artifact.joblib"
    pipelined_model = deserialize_estimator_pipeline(artifact_path)
    return pipelined_model
