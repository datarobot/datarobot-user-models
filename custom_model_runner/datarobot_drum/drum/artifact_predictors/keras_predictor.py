import pandas as pd

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
    TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


class KerasPredictor(ArtifactPredictor):
    def __init__(self):
        super(KerasPredictor, self).__init__(
            SupportedFrameworks.KERAS, PythonArtifacts.KERAS_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            from tensorflow.keras.models import load_model

            return True
        except ImportError:
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.KERAS]

    def can_load_artifact(self, artifact_path):
        if not self.is_artifact_supported(artifact_path):
            return False

        try:
            from tensorflow.keras.models import load_model

            return True
        except ImportError:
            return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        try:
            from sklearn.pipeline import Pipeline
            from tensorflow import keras as keras_tf

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is Keras
                if isinstance(model[-1], keras_tf.Model):
                    return True
            elif isinstance(model, keras_tf.Model):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False
        return False

    def load_model_from_artifact(self, artifact_path):
        from tensorflow.keras.models import load_model

        self._model = load_model(artifact_path, compile=False)
        return self._model

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(KerasPredictor, self).predict(data, model, **kwargs)
        predictions = model.predict(data)
        if self.target_type.value in TargetType.CLASSIFICATION.value:
            if predictions.shape[1] == 1:
                if self.target_type == TargetType.MULTICLASS:
                    raise DrumCommonException(
                        "Target type '{}' predictions must return the "
                        "probability distribution for all class labels".format(self.target_type)
                    )
                predictions = pd.DataFrame(predictions, columns=[self.positive_class_label])
                predictions[self.negative_class_label] = 1 - predictions[self.positive_class_label]
            else:
                predictions = pd.DataFrame(predictions, columns=self.class_labels)
        elif self.target_type in [TargetType.REGRESSION, TargetType.ANOMALY]:
            predictions = pd.DataFrame(predictions, columns=[REGRESSION_PRED_COLUMN])
        else:
            raise DrumCommonException(
                "Target type '{}' is not supported by '{}' predictor".format(
                    self.target_type.value, self.__class__.__name__
                )
            )

        return predictions
