"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import glob
import os
import pickle

import pandas as pd
from scipy.sparse import issparse

from datarobot_drum.custom_task_interfaces import TransformerInterface
from datarobot_drum.drum.exceptions import DrumSerializationError


class CustomTask(TransformerInterface):
    @classmethod
    def load_task(cls, artifact_directory):
        """
        Helper method to abstract deserializing the pickle object stored within `artifact_directory` and
        returning the custom task. Any variables that were excluded in `save_task` must be manually loaded
        proceeding this function.

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        cls
            The deserialized object
        """
        # only one artifact is expected in the folder: sklearn_transform.pkl or sklearn_transform_dense.pkl
        artifact_file = glob.glob(os.path.join(artifact_directory, "transform_sparse.pkl"))[0]
        with open(artifact_file, "rb") as fp:
            deserialized_object = pickle.load(fp)

        if not isinstance(deserialized_object, cls):
            raise DrumSerializationError(
                "load_task method must return a {} class".format(cls.__name__)
            )
        return deserialized_object

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: pd.DataFrame - training data to perform transform on
        transformer: object - trained transformer object
        y: pd.Series (optional) - target data to perform transform on
        Returns
        -------
        transformed DataFrame resulting from applying transform to incoming data
        """
        assert all(col.lower().startswith("a") for col in X.columns)
        transformed = self.transformer.transform(X)
        if issparse(transformed):
            return pd.DataFrame.sparse.from_spmatrix(
                transformed, columns=[f"feature_{i}" for i in range(transformed.shape[1])]
            )
        else:
            return pd.DataFrame(transformed)
