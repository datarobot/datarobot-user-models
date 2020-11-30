import pandas as pd


def transform(data, transformer):
    """
    Parameters
    ----------
    data: pd.DataFrame - training data to perform transform on
    transformer: object - trained transformer object
    Returns
    -------
    transformed DataFrame resulting from applying transform to incoming data
    """
    return pd.DataFrame(transformer.transform(data))
