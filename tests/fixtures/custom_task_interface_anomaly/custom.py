"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom estimator task implements anomaly detection using OneClassSVM

import pandas as pd
from sklearn.ensemble import IsolationForest

from datarobot_drum.custom_task_interfaces import AnomalyEstimatorInterface


class CustomTask(AnomalyEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        assert y is None
        self.estimator = IsolationForest()
        self.estimator.fit(X)

        return self

    def predict(self, X, **kwargs):

        # Note how anomaly estimators only use one column, so no explicit column names are needed
        return pd.DataFrame(data=self.estimator.predict(X))
