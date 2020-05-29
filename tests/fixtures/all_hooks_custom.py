import pandas as pd

prediction_value = None


def init(**kwargs):
    global prediction_value
    prediction_value = 1


def load_model(input_dir):
    global prediction_value
    prediction_value += 1
    return "dummy"


def transform(data, model):
    global prediction_value
    prediction_value += 1
    return data


def score(data, model, **kwargs):
    global prediction_value
    prediction_value += 1
    predictions = pd.DataFrame(
        [prediction_value for _ in range(data.shape[0])], columns=["Predictions"]
    )
    return predictions


def post_process(predictions, model):
    return predictions + 1
