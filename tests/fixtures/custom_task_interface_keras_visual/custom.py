"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import pandas as pd

from model_utils import (
    fit_image_classifier_pipeline,
    serialize_estimator_pipeline,
    deserialize_estimator_pipeline,
)

from datarobot_drum.custom_task_interfaces import BinaryEstimatorInterface


class CustomTask(BinaryEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        self.estimator = fit_image_classifier_pipeline(X, y, kwargs.get("class_order"))
        return self

    def save(self, artifact_directory):
        """Note how we use 2 separate serialization methods: a custom one in the helper function to store
        the keras estimator in self.estimator, and then a standard pickle to store the CustomTask object
        """
        # Because we use a pipeline to convert our images from base64 to pixel values
        # for our neural network, we need our own serialization strategy
        serialize_estimator_pipeline(self.estimator, artifact_directory)

        # Helper method to handle serializing, via pickle, the CustomTask class
        self.save_task(artifact_directory, exclude=["estimator"])

        return self

    @classmethod
    def load(cls, artifact_directory):
        """Note how we load the serialized objects in the reverse order of the save function above.
        First we load the pickle to instantiate the CustomTask object, then load the keras model
        using the deserialize helper function and attach it to custom_task.estimator.

        Also note this is a class method so we attach everything to custom_task which is automatically
        coverted back to an object that is passed to the predict hook below in self
        """
        # Helper method to load the serialized CustomTask class
        custom_task = cls.load_task(artifact_directory)
        custom_task.estimator = deserialize_estimator_pipeline(artifact_directory)

        return custom_task

    def predict_proba(self, X, **kwargs):
        """Note that for binary problems we need to specify the positive and negative class labels"""
        predictions = self.estimator.predict(X)

        # Note that binary estimators require two columns in the output, the positive and negative class labels
        # So we need to pass in the the class names derived from the estimator as column names OR
        # we can use the class labels from DataRobot stored in
        # kwargs['positive_class_label'] and kwargs['negative_class_label']
        predictions_df = pd.DataFrame(predictions, columns=[kwargs["positive_class_label"]])
        predictions_df[kwargs["negative_class_label"]] = (
            1 - predictions_df[kwargs["positive_class_label"]]
        )

        return predictions_df
