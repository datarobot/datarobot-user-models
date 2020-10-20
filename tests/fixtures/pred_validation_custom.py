import pandas as pd


def transform(data, model):
    if "MEDV" in data:
        data.pop("MEDV")
    if "Species" in data:
        data.pop("Species")
    if "class" in data:
        data.pop("class")
    data = data.fillna(0)
    return data


def post_process(predictions, model):
    return predictions
