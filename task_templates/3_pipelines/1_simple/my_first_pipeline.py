"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from datarobot_drum import drum_autofit

numeric_transformer = ColumnTransformer(
    transformers=[
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
            make_column_selector(dtype_include=np.number),
        )
    ]
)
pipeline = Pipeline(steps=[("numeric", numeric_transformer), ("model", Ridge())])


# The drum_autofit function will mark the pipeline object so that DRUM
# knows that this is the object you want to use to train your model
drum_autofit(pipeline)
