import pandas as pd
from scipy.sparse.csr import csr_matrix


def transform(X, transformer, y=None):
    """
    Parameters
    ----------
    X: pd.DataFrame - training data to perform transform on
    transformer: object - trained transformer object
    y: pd.Series (optional) - target data to perform transform on
    Returns
    -------
    transformed DataFrame resulting from applying transform to incoming data
    """
    transformed = transformer.transform(X)
    if type(transform) == csr_matrix:
        return pd.DataFrame.sparse.from_spmatrix(transformed), y
    else:
        return pd.DataFrame(transformed), y
