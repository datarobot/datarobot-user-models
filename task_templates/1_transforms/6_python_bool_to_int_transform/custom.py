"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import Any

import numpy as np
import pandas as pd

from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """This hook defines how DataRobot will train this task. Even transform tasks need to be trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def transform_bools(values: pd.Series) -> pd.Series:
        if values.dtype == np.bool:
            return values.astype(np.int)
        else:
            return values

    def transform(self, data: pd.DataFrame) -> None:
        """This hook defines how DataRobot will use the trained object from fit() to transform new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the transformed data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        data: pd.DataFrame
            Data that DataRobot passes for transformation.


        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """

        return data.apply(self.transform_bools)
