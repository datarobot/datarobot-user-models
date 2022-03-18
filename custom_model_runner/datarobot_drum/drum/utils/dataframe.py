import pandas as pd


def is_sparse_dataframe(dataframe: pd.DataFrame) -> bool:
    return hasattr(dataframe, "sparse")
