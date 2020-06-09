import pandas as pd


def transform(data, model):
    """
    Note: This hook may not have be implemented for your model.
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
    if "MEDV" in data:
        data.pop("MEDV")
    if "Species" in data:
        data.pop("Species")
    data = data.fillna(0)
    return data
