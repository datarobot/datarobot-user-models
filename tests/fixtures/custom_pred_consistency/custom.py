"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import os
import io
from typing import List, Optional

g_code_dir = None


def init(code_dir):
    """
    bookeeping to set code dir so we can separately dump preprocessing and model
    """
    global g_code_dir
    g_code_dir = code_dir


def read_input_data(input_binary_data):
    """
    read input data. Drops diag_1_desc if present,
     sets global input filename to filename passed to this function
    """
    data = pd.read_csv(io.BytesIO(input_binary_data))
    try:
        data.drop(["diag_1_desc"], axis=1, inplace=True)
    except:
        pass

    return data


def create_preprocessing_pipeline(X):
    """
    perform preprocessing on X:
        - drop diag_1_desc
        - convert all diag vars to string
        - convert race of obj
        - median impute and scale numeric features
        - constant impute and one-hot encode categorical features
    """
    # Drop diag_1_desc columns
    try:
        X.drop(["diag_1_desc"], axis=1, inplace=True)
    except:
        pass

    X["race"] = X["race"].astype("object")
    X["diag_1"] = X["diag_1"].astype("str")
    X["diag_2"] = X["diag_2"].astype("str")
    X["diag_3"] = X["diag_3"].astype("str")

    # Preprocessing for numerical features
    numeric_features = list(X.select_dtypes("int64").columns)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Preprocessing for categorical features
    categorical_features = list(X.select_dtypes("object").columns)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Preprocessor with all of the steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full preprocessing pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir=str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """
    Fit preprocessing pipeline and model.
    Dump as two separate pickle files to output dir.
    """
    pipeline = create_preprocessing_pipeline(X)
    # Train the model-Pipeline
    pipeline.fit(X, y)

    # Preprocess x
    preprocessed = pipeline.transform(X)
    preprocessed = pd.DataFrame.sparse.from_spmatrix(preprocessed)

    model = LogisticRegression(solver="liblinear")
    model.fit(preprocessed, y)

    joblib.dump(pipeline, "{}/preprocessing.pkl".format(output_dir))
    joblib.dump(model, "{}/model.pkl".format(output_dir))


def score(data, model, **kwargs):
    """
    Specifically grab the path of preprocessing pipeline, using global code dir set earlier
    Same preprocessing as performed above: change some data types, run the pipeline, and return
    the transformed data

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """

    # Make sure data types are correct for my multi-type columns.
    data["race"] = data["race"].astype("object")
    data["diag_1"] = data["diag_1"].astype("str")
    data["diag_2"] = data["diag_2"].astype("str")
    data["diag_3"] = data["diag_3"].astype("str")

    pipeline_path = "preprocessing.pkl"
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    transformed = pipeline.transform(data)
    data = pd.DataFrame.sparse.from_spmatrix(transformed)
    model_path = "model.pkl"
    model = joblib.load(os.path.join(g_code_dir, model_path))
    pred = model.predict_proba(data)
    return pd.DataFrame(pred, columns=model.classes_)


def load_model(code_dir):
    """
    We need this hook because we're dumping two pickle files: the model
    and the preprocessing pipeline, so need to tell DRUM which is the model
    """
    model_path = "model.pkl"
    model = joblib.load(os.path.join(code_dir, model_path))
    return model
