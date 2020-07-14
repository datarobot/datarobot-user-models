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


# The custom function will tag the pipeline object so that DRUM knows that this object
# is the one you want to use to train your model with
drum_autofit(pipeline)
