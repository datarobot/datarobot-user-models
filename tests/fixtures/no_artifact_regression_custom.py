import pandas as pd


def load_model(input_dir):
    return "dummy"


def transform(data, model):
    data = data.fillna(0)
    return data


def score(data, model, **kwargs):
    predictions = pd.DataFrame([1.0] * data.shape[0], columns=["Predictions"])
    return predictions
