from sklearn.pipeline import Pipeline

from model_utils import deserialize_estimator_pipeline, get_imputation_img


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

    # for Loan Lending Club dataset
    if "class" in data:
        data.pop("class")

    data = data.fillna(get_imputation_img())
    return data
