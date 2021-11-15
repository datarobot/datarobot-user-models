"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import numpy as np


class MissingValuesMedianImputation:
    """
    Custom class used to define a custom transform that imputes missing values with a median:
    In fit(), it computes the medians for all numeric features
    In transform(), it imputes missing values for all numeric features and passes the transformed data
    """

    def __init__(self, X):
        self.medians = {}

    def fit(self, X, y=None, **kwargs):
        self.medians = X.median(axis=0, numeric_only=True, skipna=True).to_dict()
        return self

    def transform(self, X):
        return X.fillna(self.medians)
