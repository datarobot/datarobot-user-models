import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression  # DELETE

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import os
from typing import List, Optional

g_input_filename = None
g_code_dir = None


def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir


def read_input_data(input_filename):
    data = pd.read_csv(input_filename)
    try:
        data.drop(['diag_1_desc'], axis=1, inplace=True)
    except:
        pass

    # Saving this for later
    global g_input_filename
    g_input_filename = input_filename
    return data


def fit(
        X: pd.DataFrame,
        y: pd.Series,
        output_dir=str,
        class_order: Optional[List[str]] = None,
        row_weights: Optional[np.ndarray] = None,
        **kwargs,
) -> None:
    # Drop diag_1_desc columns
    try:
        X.drop(['diag_1_desc'], axis=1, inplace=True)
    except:
        pass

    X['race'] = X['race'].astype('object')
    X['diag_1'] = X['diag_1'].astype('str')
    X['diag_2'] = X['diag_2'].astype('str')
    X['diag_3'] = X['diag_3'].astype('str')

    # Preprocessing for numerical features
    numeric_features = list(X.select_dtypes('int64').columns)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical features
    categorical_features = list(X.select_dtypes('object').columns)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Preprocessor with all of the steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Full preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Train the model-Pipeline
    pipeline.fit(X, y)

    # Preprocess x
    preprocessed = pipeline.transform(X)
    preprocessed = pd.DataFrame.sparse.from_spmatrix(preprocessed)

    model = LogisticRegression()
    model.fit(preprocessed, y)

    joblib.dump(pipeline, '{}/preprocessing.pkl'.format(output_dir))
    joblib.dump(model, '{}/model.pkl'.format(output_dir))


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """

    # Make sure data types are correct for my multi-type columns.
    data['race'] = data['race'].astype('object')
    data['diag_1'] = data['diag_1'].astype('str')
    data['diag_2'] = data['diag_2'].astype('str')
    data['diag_3'] = data['diag_3'].astype('str')

    pipeline_path = 'preprocessing.pkl'
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    transformed = pipeline.transform(data)
    data = pd.DataFrame.sparse.from_spmatrix(transformed)

    return data


def load_model(code_dir):
    model_path = 'model.pkl'
    model = joblib.load(os.path.join(code_dir, model_path))
    return model


def score(data, model, **kwargs):
    # doctored so predictions are random (but still sum to one)
    results = model.predict_proba(data) + np.random.randn()
    predictions = pd.DataFrame({'True': results[:, 0], 'False': 1 - results[:, 0]})

    return predictions
