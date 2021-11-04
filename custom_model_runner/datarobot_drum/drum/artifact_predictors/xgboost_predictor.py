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


class XGBoostPredictor(ArtifactPredictor):
    """
    This Predictor supports both XGBoost native & sklearn api wrapper as well
    """

    def __init__(self):
        super(XGBoostPredictor, self).__init__(
            SupportedFrameworks.XGBOOST, PythonArtifacts.PKL_EXTENSION
        )

    def is_framework_present(self):
        try:
            import xgboost

            return True
        except ImportError as e:
            self._logger.debug("Got error in imports: {}".format(e))
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.XGBOOST]

    def can_load_artifact(self, artifact_path):
        if self.is_artifact_supported(artifact_path) and self.is_framework_present():
            return True
        return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False
        try:
            from sklearn.pipeline import Pipeline
            import xgboost

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is XGBoost
                if isinstance(
                    model[-1], (xgboost.sklearn.XGBClassifier, xgboost.sklearn.XGBRegressor)
                ):
                    return True
            elif isinstance(model, xgboost.core.Booster):
                return True
            return False
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False

    def load_model_from_artifact(self, artifact_path):
        with open(artifact_path, "rb") as picklefile:
            try:
                model = pickle.load(picklefile, encoding="latin1")
            except TypeError:
                model = pickle.load(picklefile)
            return model

    def predict(self, data, model, **kwargs):
        super(XGBoostPredictor, self).predict(data, model, **kwargs)

        import xgboost

        xgboost_native = False
        if isinstance(model, xgboost.core.Booster):
            xgboost_native = True
            data = xgboost.DMatrix(data)

        labels_to_use = None
        if hasattr(model, "classes_"):
            labels_to_use = model.classes_
        if self.target_type.value in TargetType.CLASSIFICATION.value:
            if xgboost_native:
                predictions = model.predict(data)
            else:
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
