import numpy as np
import pyarrow as pa

from io import BytesIO, StringIO
from scipy.sparse.csr import csr_matrix

MEGABYTE = 1024 * 1024
CUSTOM_TRANSFORM_OUTPUT_MAX_CHUNK_SIZE = 5 * MEGABYTE


def is_sparse(df):
    return hasattr(df, 'sparse') or type(df.iloc[0].values[0]) == csr_matrix


# Dense dataframe handling
def _create_chunks_dense(df, num_splits):
    return np.array_split(df, num_splits)


# TODO: make this private, implement chunking to deal with bigger datasets (?)
def make_arrow_payload(chunk):
    sink = BytesIO()
    batch = pa.RecordBatch.from_pandas(chunk)
    with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def read_arrow_payload(sink):
    # TODO: this doesn't work
    buf = sink.getvalue()
    df = pa.ipc.open_stream(buf).read_pandas()
    return df


def _get_dense_data_size(df):
    return df.memory_usage(index=False, deep=True).sum()



