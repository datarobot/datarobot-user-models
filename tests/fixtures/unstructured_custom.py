def load_model(input_dir):
    return "dummy"


def score_unstructured(model, **kwargs):
    print(kwargs)
    in_data = kwargs.get("data")

    if isinstance(in_data, bytes):
        in_data = in_data.decode("utf8")

    words_count = in_data.count(" ") + 1

    ret_mode = kwargs.get("ret_mode", "")
    if ret_mode == "binary":
        ret = {
            "data": words_count.to_bytes(4, byteorder="big"),
            "mimetype": "application/octet-stream",
        }
    else:
        ret = {"data": str(words_count)}
    return ret
