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
        """ This hook defines how DataRobot will train this task. Even transform tasks need to be trained to learn/store information from training data
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containg a trained object [in this example - median of each numeric column], that is then used to transform new data.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column (None is passed for unsupervised projects).

        Returns
        -------
        CustomTask
            returns an object instance of class CustomTask that can be used in chained method calls
        """
        pass

    @staticmethod
    def _transform_bools(values: pd.Series) -> pd.Series:
        """ Helper method example. This could also be placed in a separate file as well.
        """
        if values.dtype == np.bool:
            return values.astype(np.int)
        else:
            return values

    def transform(self, X, **kwargs):
        """ This hook defines how DataRobot will use the trained object from fit() to transform new data.
        DataRobot runs this hook when the task is used for scoring inside a blueprint.
        As an output, this hook is expected to return the transformed data.
        The input parameters are passed by DataRobot based on dataset and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Data that DataRobot passes for transformation.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with transformed data.
        """

        return X.apply(self._transform_bools)