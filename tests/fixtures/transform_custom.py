import pandas as pd
from scipy.sparse.csr import csr_matrix


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
    transformed = transformer.transform(data)
    if type(transform) == csr_matrix:
        return pd.DataFrame.sparse.from_spmatrix(transformed)
    else:
        return pd.DataFrame(transformed)
