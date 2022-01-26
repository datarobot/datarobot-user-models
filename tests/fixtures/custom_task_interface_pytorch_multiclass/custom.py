"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from pathlib import Path
from typing import List, Optional, Any, Dict
import pandas as pd
import torch
import os
from datarobot_drum.custom_task_interfaces import MulticlassEstimatorInterface
from sklearn.preprocessing import LabelEncoder
import pickle

from model_utils import (
    build_classifier,
    train_classifier,
)


class CustomTask(MulticlassEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        """Note how we encode the class labels and store them on self to be used in the predict hook"""
        self.lb = LabelEncoder().fit(y)
        y = self.lb.transform(y)

        # For reproducible results
        torch.manual_seed(0)

        self.estimator, optimizer, criterion = build_classifier(X, len(self.lb.classes_))
        train_classifier(X, y, self.estimator, optimizer, criterion)

    def save(self, artifact_directory):
        """
        Serializes the object and stores it in `artifact_directory`

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        self
        """

        # If your estimator is not pickle-able, you can serialize it using its native method,
        # i.e. in this case for keras we use model.save, and then set the estimator to none
        torch.save(self.estimator, Path(artifact_directory) / "torch_class.pth")
        self.estimator = None

        # Now that the estimator is none, it won't be pickled with the CustomTask class (i.e. this one)
        with open(Path(artifact_directory) / "artifact.pkl", "wb") as fp:
            pickle.dump(self, fp)

        return self

    @classmethod
    def load(cls, artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`

        Returns
        -------
        cls
            The deserialized object
        """
        with open(Path(artifact_directory) / "artifact.pkl", "rb") as fp:
            custom_task = pickle.load(fp)

        custom_task.estimator = torch.load(Path(artifact_directory) / "torch_class.pth")
        return custom_task

    def predict_proba(self, X, **kwargs):
        """Note how the column names come from the encoded class labels in the fit hook above"""
        data_tensor = torch.from_numpy(X.values).type(torch.FloatTensor)
        predictions = self.estimator(data_tensor).cpu().data.numpy()

        return pd.DataFrame(data=predictions, columns=self.lb.classes_)
