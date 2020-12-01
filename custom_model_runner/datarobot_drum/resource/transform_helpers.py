import pyarrow as pa
import pandas as pd

from io import BytesIO, StringIO

from scipy.io import mmwrite, mmread
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack

from datarobot_drum.drum.common import X_TRANSFORM_KEY


def is_sparse(df):
    return hasattr(df, "sparse") or type(df.iloc[0].values[0]) == csr_matrix


def make_arrow_payload(df):
    return pa.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()


def make_csv_payload(df):
    s_buf = StringIO()
    df.to_csv(s_buf, index=False)
    return s_buf.getvalue().encode("utf-8")


def read_arrow_payload(response_dict):
    bytes = response_dict[X_TRANSFORM_KEY]
    df = pa.ipc.deserialize_pandas(bytes)
    return df


def read_csv_payload(response_dict):
    bytes = response_dict[X_TRANSFORM_KEY]
    return pd.read_csv(BytesIO(bytes))


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
