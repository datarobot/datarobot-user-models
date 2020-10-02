import pandas as pd


def load_model(input_dir):
    return "dummy"


def score_unstructured(model, **kwargs):
    in_data = kwargs.get("data")
    ret = {}
    if isinstance(in_data, bytes):
        in_data = in_data.decode("utf8")

    if "ret_mimetype" in kwargs:
        ret["mimetype"] = kwargs["ret_mimetype"]
    if "ret_charset" in kwargs:
        ret["charset"] = kwargs["ret_charset"]
    ret["data"] = kwargs["ret_text"]
    return ret
