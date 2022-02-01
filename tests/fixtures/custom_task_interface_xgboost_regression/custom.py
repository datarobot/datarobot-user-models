"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd

from create_pipeline import make_regressor_pipeline
from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        """Note how this estimator actually uses a helper function to build an sklearn pipeline to
        drop all non-numeric values, impute the mean, etc. We generally recommend though that
        tasks are separated to promote reuse, i.e. the transformations in the below pipline could be placed in their own
        Custom Tasks as transforms.
        """
        self.estimator = make_regressor_pipeline(X)
        self.estimator.fit(X, y)

    def predict(self, X, **kwargs):
        # Note how the regression estimator only outputs one column, so no explicit column names are needed
        return pd.DataFrame(data=self.estimator.predict(X))
