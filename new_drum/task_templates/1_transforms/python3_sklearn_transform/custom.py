"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pickle
import pandas as pd
from scipy.sparse.csr import csr_matrix

from create_transform_pipeline import make_pipeline
from new_drum.src.transform import TransformerInterface


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
        self.transformer = make_pipeline()

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: pd.DataFrame - training data to perform transform on

        Returns
        -------
        transformed DataFrame resulting from applying transform to incoming data
        """
        transformed = self.transformer.transform(X)

        if type(transformed) == csr_matrix:
            return pd.DataFrame.sparse.from_spmatrix(transformed)
        else:
            return pd.DataFrame(transformed)
