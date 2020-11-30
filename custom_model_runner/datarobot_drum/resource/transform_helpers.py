import numpy as np
import pyarrow as pa

from io import BytesIO, StringIO
from scipy.sparse.csr import csr_matrix

MEGABYTE = 1024 * 1024
CUSTOM_TRANSFORM_OUTPUT_MAX_CHUNK_SIZE = 5 * MEGABYTE


def is_sparse(df):
    return hasattr(df, 'sparse') or type(df.iloc[0].values[0]) == csr_matrix


# Dense dataframe handling
def make_arrow_payload(df):
    return pa.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()


def read_arrow_payload(response_dict):
    bytes = response_dict['transformations']
    df = pa.ipc.deserialize_pandas(bytes)
    return df


def _get_dense_data_size(df):
    return df.memory_usage(index=False, deep=True).sum()



