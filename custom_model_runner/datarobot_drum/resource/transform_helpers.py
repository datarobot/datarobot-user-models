import pandas as pd
import logging

from cgi import FieldStorage
from io import BytesIO, StringIO

from scipy.io import mmwrite, mmread
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack

from datarobot_drum.drum.common import verify_pyarrow_module


def filter_urllib3_logging():
    """Filter header errors from urllib3 due to a urllib3 bug."""
    urllib3_logger = logging.getLogger("urllib3.connectionpool")
    if not any(isinstance(x, NoHeaderErrorFilter) for x in urllib3_logger.filters):
        urllib3_logger.addFilter(NoHeaderErrorFilter())


class NoHeaderErrorFilter(logging.Filter):
    """Filter out urllib3 Header Parsing Errors due to a urllib3 bug."""

    def filter(self, record):
        """Filter out Header Parsing Errors."""
        return "Failed to parse headers" not in record.getMessage()


def is_sparse(df):
    return hasattr(df, "sparse") or type(df.iloc[0].values[0]) == csr_matrix


def make_arrow_payload(df, arrow_version):
    pa = verify_pyarrow_module()

    if arrow_version != pa.__version__ and arrow_version < 0.2:
        batch = pa.RecordBatch.from_pandas(df, nthreads=None, preserve_index=False)
        sink = pa.BufferOutputStream()
        options = pa.ipc.IpcWriteOptions(
            metadata_version=pa.MetadataVersion.V4, use_legacy_format=True
        )
        with pa.RecordBatchStreamWriter(sink, batch.schema, options=options) as writer:
            writer.write_batch(batch)
        return sink.getvalue().to_pybytes()
    else:
        return pa.ipc.serialize_pandas(df, preserve_index=False).to_pybytes()


def make_csv_payload(df):
    s_buf = StringIO()
    df.to_csv(s_buf, index=False)
    return s_buf.getvalue().encode("utf-8")


def read_arrow_payload(response_dict, transform_key):
    pa = verify_pyarrow_module()

    bytes = response_dict[transform_key]
    df = pa.ipc.deserialize_pandas(bytes)
    return df


def read_csv_payload(response_dict, transform_key):
    bytes = response_dict[transform_key]
    return pd.read_csv(BytesIO(bytes))


def make_mtx_payload(df):
    if hasattr(df, "sparse"):
        sparse_mat = csr_matrix(df.sparse.to_coo())
    else:
        sparse_mat = vstack(x[0] for x in df.values)
    sink = BytesIO()
    mmwrite(sink, sparse_mat)
    return sink.getvalue()


def read_mtx_payload(response_dict, transform_key):
    bytes = response_dict[transform_key]
    sparse_mat = mmread(BytesIO(bytes))
    return csr_matrix(sparse_mat)


def validate_transformed_output(transformed_output, should_be_sparse=False):
    if should_be_sparse:
        assert type(transformed_output) == csr_matrix
        assert transformed_output.shape[1] == 714
    else:
        assert type(transformed_output) == pd.DataFrame
        assert transformed_output.shape[1] == 10


def parse_multi_part_response(response):
    parsed_response = {}
    fs = FieldStorage(
        fp=BytesIO(response.content),
        headers=response.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": response.headers["Content-Type"],
        },
    )
    for child in fs.list:
        key = child.name
        value = child.file.read()
        parsed_response.update({key: value})

    return parsed_response
