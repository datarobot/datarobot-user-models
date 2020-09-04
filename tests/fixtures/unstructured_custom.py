import pandas as pd


def load_model(input_dir):
    return "dummy"


def read_input_data(input_filename):
    with open(input_filename) as f:
        data = f.read()
    return data


def score(data, model, **kwargs):
    return str(data.count(" ") + 1)
