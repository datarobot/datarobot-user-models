prediction_value = None


def init(**kwargs):
    global prediction_value
    prediction_value = 1


def load_model(input_dir):
    global prediction_value
    prediction_value += 1
    return "dummy"


def score_unstructured(model, data, **kwargs):
    global prediction_value
    prediction_value += 1
    return str(prediction_value)
