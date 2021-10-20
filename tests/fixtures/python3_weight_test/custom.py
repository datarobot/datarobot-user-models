"""
Task to verify that sample weights are handled correctly
"""
import pickle
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[pd.DataFrame] = None,
    **kwargs,
):
    estimator = LogisticRegression()
    np.testing.assert_allclose(row_weights.to_numpy().ravel(), X["weight_check"])
    estimator.fit(X, y)

    # You must serialize out your model to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(estimator, fp)


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(data=model.predict_proba(data), columns=model.classes_)
