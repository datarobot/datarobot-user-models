"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom estimator task implements a decision tree regressor

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface


class CustomTask(RegressionEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        self.estimator = DecisionTreeRegressor()
        self.estimator.fit(X, y)

        return self

    def predict(self, X, **kwargs):
        return pd.DataFrame(data=self.estimator.predict(X))
