"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from pathlib import Path
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from datarobot_drum.custom_task_interfaces import BinaryEstimatorInterface

from model_utils import (
    build_classifier,
    train_classifier,
)


class CustomTask(BinaryEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        self.lb = LabelEncoder().fit(y)
        y = self.lb.transform(y)

        # For reproducible results
        torch.manual_seed(0)
        self.estimator, self.optimizer, self.criterion = build_classifier(X, len(self.lb.classes_))
        train_classifier(X, y, self.estimator, self.optimizer, self.criterion)
        return self

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
        # i.e. in this case for pytorch we use model.save, and then set the estimator to none
        torch.save(self.estimator, Path(artifact_directory) / "torch_class.pth")

        # Helper method to handle serializing, via pickle, the CustomTask class
        self.save_task(artifact_directory, exclude=["estimator"])

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
        custom_task = cls.load_task(artifact_directory)
        custom_task.estimator = torch.load(Path(artifact_directory) / "torch_class.pth")

        return custom_task

    def predict_proba(self, X, **kwargs):
        """Since pytorch only outputs a single probability, i.e. the probability of the positive class,
        we use the class labels passed in kwargs to label the columns"""
        data_tensor = torch.from_numpy(X.values).type(torch.FloatTensor)
        predictions = self.estimator(data_tensor).cpu().data.numpy()

        # Note that binary estimators require two columns in the output, the positive and negative class labels
        # So we need to pass in the the class names derived from the estimator as column names OR
        # we can use the class labels from DataRobot stored in
        # kwargs['positive_class_label'] and kwargs['negative_class_label']
        predictions = pd.DataFrame(predictions, columns=[kwargs["positive_class_label"]])

        # The negative class probability is just the inverse of what the model predicts above
        predictions[kwargs["negative_class_label"]] = (
            1 - predictions[kwargs["positive_class_label"]]
        )
        return predictions
