"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
# This custom transform task implements missing values imputation using a median

from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    def fit(self, X, y, **kwargs):
        # compute medians for all numeric features on training data, store them in a dictionary
        self.fit_data = X.median(axis=0, numeric_only=True, skipna=True).to_dict()
        return self

    def transform(self, X, **kwargs):
        return X.fillna(self.fit_data)
