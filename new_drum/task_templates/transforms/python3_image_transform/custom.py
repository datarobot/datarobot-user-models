"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import Union
import pandas as pd

from img_utils import (
    b64_to_img,
    img_to_b64,
    img_to_grayscale,
)
from new_drum.src.transform import TransformerInterface


class CustomTask(TransformerInterface):
    @staticmethod
    def _process_image(raw_data: Union[str, bytes]) -> bytes:
        img = b64_to_img(raw_data)
        img = img_to_grayscale(img)
        return img_to_b64(img)

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

        return X.applymap(self._process_image)
