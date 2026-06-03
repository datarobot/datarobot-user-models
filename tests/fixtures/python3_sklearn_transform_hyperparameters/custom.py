"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.custom_task_interfaces import TransformerInterface

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer


class CustomTask(TransformerInterface):
    def fit(self, X, y, parameters=None, **kwargs):
        """This numeric transform example showcases how to define and use hyperparameters. Hyperparameters are defined
        in model-metadata.yaml, and are passed into fit as a dict.

        This transformer computes missing values using Sklearn's SimpleImputer, then transforms the output of that into
        bins using KBinsDiscretizer.

        Parameters
        -------
        X: pd.DataFrame
            Training data
        y: pd.Series
            Project's target column.
        parameters: dict[str, Any] or None
            Dict containing mapping of parameter names and values. The parameter value's type can be an int, float, or
            str depending on the parameter definition's type in model-metadata.yaml.

            In this specific example, the parameters are:
                seed: int
                    Numpy seed that gets set for reproducible fit behavior.
                missing_values_strategy: (multi) str or float
                    Strategy for imputing missing values. Can be one of {'mean', 'median', 'most_frequent'} or a float.
                    If it's a float, we set the SimpleImputer strategy to be constant, and use the float as the fill
                    value.
                kbins_n_bins: int
                    Number of bins to produce.
                kbins_strategy: (select) str
                    Strategy for binning. Can be one of {'uniform', 'quantile', 'kmeans'}.
                print_message: (string) str
                    A string parameter showcasing that you can pass arbitrary strings as parameters. In this example, we
                    simply print it, but this can be used for practically anything such as passing JSON, a list, dict,
                    etc as a string, then parsing it in your custom task.

            See how they are defined in model-metadata.yaml.

        Returns
        -------
        CustomTask
        """
        # Parameters should always be passed in as a dict if hyperparameters are defined in the model-metadata.
        if parameters is None:
            raise ValueError(
                "Parameters were not passed into the custom task. "
                "Ensure your model metadata contains hyperparameter definitions"
            )

        # A useless parameter show-casing you can pass arbitrary strings
        print(parameters["print_message"])

        # Set the seed for reproducible fit behavior.
        np.random.seed(parameters["seed"])

        # First fit the SimpleImputer. Check the missing_values parameter to see if it's a numeric, if so, use the
        # 'constant' strategy and set the fill value to the numeric.
        missing_values_fill_value = None
        if isinstance(parameters["missing_values_strategy"], (int, float)):
            missing_values_strategy = "constant"
            missing_values_fill_value = parameters["missing_values_strategy"]
        else:
            missing_values_strategy = parameters["missing_values_strategy"]

        self.missing_vals_transformer = SimpleImputer(
            strategy=missing_values_strategy,
            fill_value=missing_values_fill_value,
        ).fit(X, y)

        # Then, we need to transform the data to use as training data for the second transformer.
        X = self.missing_vals_transformer.transform(X)

        # Now we fit the second transformer, and use hyperparameters to choose the number of bins and strategy.
        self.kbins_transformer = KBinsDiscretizer(
            n_bins=parameters["kbins_n_bins"],
            strategy=parameters["kbins_strategy"],
            encode="onehot",
        )
        self.kbins_transformer.fit(X, y)

        # Both transformers are fit, so now we can use them in the transform function defined below.
        return self

    def transform(self, X, **kwargs):
        # Impute the missing values
        X = self.missing_vals_transformer.transform(X)
        # Then bin them
        X = self.kbins_transformer.transform(X)
        return pd.DataFrame.sparse.from_spmatrix(X)
