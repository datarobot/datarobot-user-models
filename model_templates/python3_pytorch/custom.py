def transform(data, model):
    """
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
    # Remove target columns if they're in the dataset
    for target_col in ["Grade 2014", "Species"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)
    return data
