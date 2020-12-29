import pickle
import pandas as pd

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=np.object)


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Sparse preprocessing pipeline, for models such as Ridge that handle sparse input well
sparse_preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_selector),
        ("cat", categorical_pipeline, categorical_selector),
    ]
)


def make_pipeline():
    return sparse_preprocessing_pipeline


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models. For custom transformers, the
    transform hook below is also required.

    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame - training data to perform fit on
    y: pd.Series - target data to perform fit on
    output_dir: the path to write output. This is the path provided in '--output' parameter of the
        'drum fit' command.
    kwargs: Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    transformer = make_pipeline()
    transformer.fit(X, y)

    # You must serialize out your transformer to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model or transformer
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)
