"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pickle

from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor
from datarobot_drum.drum.enum import extra_deps, PythonArtifacts, SupportedFrameworks, TargetType
from datarobot_drum.drum.exceptions import DrumCommonException


class SKLearnPredictor(ArtifactPredictor):
    def __init__(self):
        super(SKLearnPredictor, self).__init__(
            SupportedFrameworks.SKLEARN, PythonArtifacts.PKL_EXTENSION
        )

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.SKLEARN]

    def is_framework_present(self):
        try:
            from sklearn.base import BaseEstimator

            return True
        except ImportError as e:
            return False

    def can_load_artifact(self, artifact_path):
        return self.is_artifact_supported(artifact_path)

    def load_model_from_artifact(self, artifact_path):
        with open(artifact_path, "rb") as picklefile:
            try:
                model = pickle.load(picklefile, encoding="latin1")
            except TypeError:
                model = pickle.load(picklefile)
            return model

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            return True
        else:
            return False

    def predict(self, data, model, **kwargs):
        super(SKLearnPredictor, self).predict(data, model, **kwargs)

        labels_to_use = None
        if self.target_type.value in TargetType.CLASSIFICATION.value:
            if hasattr(model, "classes_"):
                labels_to_use = list(model.classes_)
            predictions = model.predict_proba(data)
        elif self.target_type in [TargetType.REGRESSION, TargetType.ANOMALY]:
            predictions = model.predict(data)
        else:
            raise DrumCommonException(
                "Target type '{}' is not supported by '{}' predictor".format(
                    self.target_type.value, self.__class__.__name__
                )
            )

        return predictions, labels_to_use
