from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.common import UnstructuredDtoKeys, PredictionServerMimetypes

CHARSET_DEFAULT = "utf8"
MIMETYPE_TEXT_DEFAULT = PredictionServerMimetypes.TEXT_PLAIN
MIMETYPE_BINARY_DEFAULT = PredictionServerMimetypes.APPLICATION_OCTET_STREAM


def _is_text_mimetype(mimetype):
    return mimetype.startswith("text/") or mimetype == PredictionServerMimetypes.APPLICATION_JSON


def _resolve_incoming_unstructured_data(in_data, mimetype, charset):
    if not isinstance(in_data, bytes):
        raise DrumCommonException("bytes data is expected, received {}".format(type(in_data)))

    # Incoming mimetype that startswith `text/` or `application/json` or "" or missing is considered "textual".
    # If user sends request with "textual" mimetype, but charset is missing, we set default charset to `utf8`.
    # If user sends request with non textual mimetype, but charset is missing, we don't pass charset param into the hook.
    ret_mimetype = mimetype if mimetype is not None and mimetype != "" else MIMETYPE_TEXT_DEFAULT

    if _is_text_mimetype(ret_mimetype):
        ret_charset = charset if charset is not None else CHARSET_DEFAULT
        ret_data = in_data.decode(ret_charset)
    else:
        ret_charset = charset
        ret_data = in_data

    return ret_data, ret_mimetype, ret_charset


def _resolve_outgoing_unstructured_data(ret_data, ret_kwargs):
    if ret_kwargs is None:
        ret_kwargs = {}

    # In this function incoming ret_data expected to be either str or bytes
    # If user returns string, but doesn't provide mimetype/charset, we set them to text/plain and utf8
    # If user returns bytes, but doesn't provide mimetype/charset, we set them to application/octet-stream and None
    if isinstance(ret_data, (str, type(None))):
        ret_mimetype = ret_kwargs.get(UnstructuredDtoKeys.MIMETYPE, MIMETYPE_TEXT_DEFAULT)
        ret_charset = ret_kwargs.get(UnstructuredDtoKeys.CHARSET, CHARSET_DEFAULT)
        if ret_data is not None:
            ret_data = ret_data.encode(ret_charset)
    else:
        ret_mimetype = ret_kwargs.get(UnstructuredDtoKeys.MIMETYPE, MIMETYPE_BINARY_DEFAULT)
        ret_charset = ret_kwargs.get(UnstructuredDtoKeys.CHARSET, None)

    return ret_data, ret_mimetype, ret_charset
