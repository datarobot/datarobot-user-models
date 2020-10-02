from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.common import UnstructuredDtoKeys

CHARSET_DEFAULT = "utf8"
MIMETYPE_DEFAULT = "text/plain"


def _is_text_mimetype(mimetype):
    return mimetype.startswith("text/") or mimetype.startswith("application/json")


def _resolve_incoming_unstructured_data(in_data, mimetype, charset):
    ret_mimetype = mimetype if mimetype is not None else MIMETYPE_DEFAULT
    ret_charset = charset if charset is not None else CHARSET_DEFAULT

    if not isinstance(in_data, bytes):
        raise DrumCommonException("bytes data is expected, received {}".format(type(in_data)))

    if _is_text_mimetype(ret_mimetype):
        ret_data = in_data.decode(ret_charset)
    else:
        ret_data = in_data

    return ret_data, ret_mimetype, ret_charset


def _resolve_outgoing_unstructured_data(unstructured_response):
    ret_data = unstructured_response.get(UnstructuredDtoKeys.DATA, None)
    ret_charset = unstructured_response.get(UnstructuredDtoKeys.CHARSET, CHARSET_DEFAULT)
    ret_mimetype = unstructured_response.get(UnstructuredDtoKeys.MIMETYPE, MIMETYPE_DEFAULT)

    if isinstance(ret_data, str):
        ret_data = ret_data.encode(ret_charset)

    return ret_data, ret_mimetype, ret_charset
