#!/usr/bin/env python3

import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


# read the training dataset

split_ratio = 0.8
prediction_threshold = 0.5

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_filename = os.path.join(cur_dir, "mlops-example-surgical-dataset.csv")

df = pd.read_csv(dataset_filename)
df.insert(0, "id", [f"{x}" for x in range(100, 100 + len(df))])

arr = df.to_numpy()

# np.random.shuffle(arr)

train_data_len = int(arr.shape[0] * split_ratio)

train_data = arr[:train_data_len, :]
train_no_label = train_data[:, :-1]
label = train_data[:, -1]
label = label.astype("int")
holdout_data = arr[train_data_len:, :]  # same as the test data
test_data = holdout_data[:, :-1]

# train-and-generate-artifacts the model
clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
model = clf.fit(train_no_label, label)

model_filename = Path(cur_dir) / "../model.pkl"
pickle.dump(model, open(model_filename, "wb"))

header_full = ",".join(df.columns)
format_full = "%s," + "%f," * (len(df.columns) - 2) + "%i"

header_no_label = ",".join(df.columns[:-1])
format_no_label = "%s," + "%f," * (len(df.columns) - 3) + "%f"

training_dataset_filename = Path(cur_dir).parent / "datasets" / "training-surgical-dataset.csv"
np.savetxt(
    training_dataset_filename,
    train_data,
    delimiter=",",
    header=header_full,
    fmt=format_full,
    comments="",
)

holdout_dataset_filename = Path(cur_dir).parent / "datasets" / "holdout-surgical-dataset.csv"
np.savetxt(
    holdout_dataset_filename,
    holdout_data,
    delimiter=",",
    header=header_full,
    fmt=format_full,
    comments="",
)

test_dataset_filename = Path(cur_dir).parent / "datasets" / "predict-request-surgical-dataset.csv"
np.savetxt(
    test_dataset_filename,
    test_data,
    delimiter=",",
    header=header_no_label,
    fmt=format_no_label,
    comments="",
)

print("Done")
