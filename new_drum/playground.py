# NEW DRUM PLAYGROUND
#
# Any changes to the custom_model_runner directory should be reinstalled to your virtualenv by running
# this in custom_model_runner/
#  > make wheel && pip install dist/*
#
# When debugging this file, also be sure to set your working directory to your local datarobot-user-models dir
import sys
import pandas as pd
import numpy as np

# Make anything from new_drum importable
sys.path.insert(0, "./new_drum")

from new_drum.task_templates.transforms.python_missing_values.custom import CustomTask


def generate_X_y(num_rows, num_cols):
    column_names = [str(i) for i in range(num_cols)]
    X = pd.DataFrame(np.random.randint(0, num_rows, size=(num_rows, num_cols)))
    X.columns = column_names

    y = pd.Series([0] * num_rows)
    return X, y


X, y = generate_X_y(100, 4)
transformer = CustomTask()
transformer.fit(X, y).save(artifact_directory=".")

transformer = CustomTask.load(".")
output = transformer.transform(X)

assert output.shape == (100, 4)
