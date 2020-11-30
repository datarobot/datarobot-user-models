import pyarrow as pa

from io import BytesIO

from scipy.io import mmwrite, mmread
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack


X_TRANSFORM_KEY = "X.transformed"


def is_sparse(df):
    return hasattr(df, "sparse") or type(df.iloc[0].values[0]) == csr_matrix


def make_arrow_payload(df):
    return pa.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()


def read_arrow_payload(response_dict):
    bytes = response_dict[X_TRANSFORM_KEY]
    df = pa.ipc.deserialize_pandas(bytes)
    return df


def make_mtx_payload(df):
    if hasattr(df, "sparse"):
        sparse_mat = csr_matrix(df.sparse.to_coo())
    else:
        sparse_mat = vstack(x[0] for x in df.values)
    sink = BytesIO()
    mmwrite(sink, sparse_mat)
    return sink.getvalue()


def read_mtx_payload(response_dict):
    bytes = response_dict[X_TRANSFORM_KEY]
    sparse_mat = mmread(BytesIO(bytes))
    return csr_matrix(sparse_mat)
