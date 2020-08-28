import pickle
import pandas as pd

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


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
        # checking if positive/negative class labels were provided
        # done in the base class
        super(SKLearnPredictor, self).predict(data, model, **kwargs)

        def _determine_positive_class_index(pos_label, neg_label):
            """Find index of positive class label to interpret predict_proba output"""

            if not hasattr(model, "classes_"):
                self._logger.warning(
                    "We were not able to verify you were using the right class labels because your estimator doesn't have a classes_ attribute"
                )
                return 1
            labels = [str(label) for label in model.classes_]
            if not all(x in labels for x in [pos_label, neg_label]):
                error_message = "Wrong class labels. Use class labels detected by sklearn model: {}".format(
                    labels
                )
                raise DrumCommonException(error_message)

            return labels.index(pos_label)

        if self.positive_class_label is not None and self.negative_class_label is not None:
            predictions = model.predict_proba(data)
            positive_label_index = _determine_positive_class_index(
                self.positive_class_label, self.negative_class_label
            )
            negative_label_index = 1 - positive_label_index
            predictions = [
                [prediction[positive_label_index], prediction[negative_label_index]]
                for prediction in predictions
            ]
            predictions = pd.DataFrame(
                predictions, columns=[self.positive_class_label, self.negative_class_label]
            )
        else:
            predictions = pd.DataFrame(
                [float(prediction) for prediction in model.predict(data)],
                columns=[REGRESSION_PRED_COLUMN],
            )

        return predictions
