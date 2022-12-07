"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from io import BytesIO
import werkzeug


def load_model(input_dir):
    return "dummy"


def score_unstructured(model, data, query, **kwargs):
    print("Model: ", model)
    print("Incoming content type params: ", kwargs)
    print("Incoming data type: ", type(data))
    print("Incoming data: ", data)

    mlops = kwargs.get("mlops")
    print(f"MLOps supported: {mlops is not None}")

    headers = kwargs.get("headers")
    print("Incoming request headers: ", headers)
    print("Incoming query params: ", query)

    if headers and "multipart/form-data" in headers.get("Content-Type"):
        # For more information refer to:
        # https://werkzeug.palletsprojects.com/en/2.2.x/http/#module-werkzeug.formparser
        environ = {
            "wsgi.input": BytesIO(data),
            "CONTENT_LENGTH": headers.get("Content-Length"),
            "CONTENT_TYPE": headers.get("Content-Type"),
            "REQUEST_METHOD": "POST",
        }
        stream, form, files = werkzeug.formparser.parse_form_data(environ)
        filestorage = files.get("filekey")
        data = filestorage.stream.read()

    if isinstance(data, bytes):
        text_data = data.decode("utf8")
    else:
        text_data = data

    text_data = text_data.strip().replace("  ", " ")
    words_count = text_data.count(" ") + 1

    ret_mode = query.get("ret_mode", "")
    if ret_mode == "binary":
        ret_data = words_count.to_bytes(4, byteorder="big")
        ret_kwargs = {"mimetype": "application/octet-stream"}
        ret = ret_data, ret_kwargs
    else:
        ret = str(words_count)
    return ret
