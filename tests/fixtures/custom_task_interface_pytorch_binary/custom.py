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
        """ This hook defines how DataRobot will train this task.
        DataRobot runs this hook when the task is being trained inside a blueprint.
        As an output, this hook is expected to create an artifact containing a trained object, that is then used to predict new data.
        The input parameters are passed by DataRobot based on project and blueprint configuration.

        Parameters
        -------
        X: pd.DataFrame
            Training data that DataRobot passes when this task is being trained.
        y: pd.Series
            Project's target column.
        row_weights: np.ndarray (optional, default = None)
            A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

        Returns
        -------
        CustomTask
            returns an object instance of class CustomTask that can be used in chained method calls
        """
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

        torch.save(self, Path(artifact_directory) / "torch_class.pth")
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
        return torch.load(Path(artifact_directory) / "torch_class.pth")

    # TODO consider putting directly positive_class_label and negative_class_label into function signature directly
    def predict_proba(self, X, **kwargs):
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

        data_tensor = torch.from_numpy(X.values).type(torch.FloatTensor)
        predictions = self.estimator(data_tensor).cpu().data.numpy()

        # TODO have helper function to add in 2 columns for binary
        # have get_positive_class_labels(kwargs) -> gets rid of magic strings
        predictions = pd.DataFrame(predictions, columns=[kwargs["positive_class_label"]])
        predictions[kwargs["negative_class_label"]] = (
            1 - predictions[kwargs["positive_class_label"]]
        )
        return predictions
