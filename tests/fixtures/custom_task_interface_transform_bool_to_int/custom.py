"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import numpy as np
import pandas as pd

from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    def fit(self, X, y, **kwargs):
        """The transform hook below is stateless and doesn't require a trained transformer, so the fit is blank"""
        pass

    @staticmethod
    def _transform_bools(values: pd.Series) -> pd.Series:
        """Helper method example. This could also be placed in a separate file as well."""
        if values.dtype == np.bool:
            return values.astype(np.int)
        else:
            return values

    def transform(self, X, **kwargs):
        """Changes bools to ints using the helper function above"""
        return X.apply(self._transform_bools)
