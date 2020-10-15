from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.common import UnstructuredDtoKeys

CHARSET_DEFAULT = "utf8"
MIMETYPE_TEXT_DEFAULT = "text/plain"
MIMETYPE_BINARY_DEFAULT = "application/octet-stream"


def _is_text_mimetype(mimetype):
    return mimetype.startswith("text/") or mimetype.startswith("application/json")


def _resolve_incoming_unstructured_data(in_data, mimetype, charset):
    ret_mimetype = mimetype if mimetype is not None and mimetype != "" else MIMETYPE_TEXT_DEFAULT
    ret_charset = charset if charset is not None else CHARSET_DEFAULT

    if not isinstance(in_data, bytes):
        raise DrumCommonException("bytes data is expected, received {}".format(type(in_data)))

    if _is_text_mimetype(ret_mimetype):
        ret_data = in_data.decode(ret_charset)
    else:
        ret_data = in_data

    return ret_data, ret_mimetype, ret_charset


def _resolve_outgoing_unstructured_data(ret_data, ret_kwargs):
    if ret_kwargs is None:
        ret_kwargs = {}
    ret_charset = ret_kwargs.get(UnstructuredDtoKeys.CHARSET, CHARSET_DEFAULT)
    ret_mimetype = ret_kwargs.get(
        UnstructuredDtoKeys.MIMETYPE,
        MIMETYPE_TEXT_DEFAULT
        if isinstance(ret_data, (str, type(None)))
        else MIMETYPE_BINARY_DEFAULT,
    )
    if isinstance(ret_data, str):
        ret_data = ret_data.encode(ret_charset)

    return ret_data, ret_mimetype, ret_charset
