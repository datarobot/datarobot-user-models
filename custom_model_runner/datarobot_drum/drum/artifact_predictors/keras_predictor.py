import pandas as pd

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
)
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


class KerasPredictor(ArtifactPredictor):
    def __init__(self):
        super(KerasPredictor, self).__init__(
            SupportedFrameworks.KERAS, PythonArtifacts.KERAS_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            import keras
            from keras.models import load_model

            return True
        except ImportError:
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.KERAS]

    def can_load_artifact(self, artifact_path):
        if not self.is_artifact_supported(artifact_path):
            return False

        try:
            import keras
            from keras.models import load_model

            return True
        except ImportError:
            return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        try:
            import keras
            from sklearn.pipeline import Pipeline

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is Keras
                if isinstance(model[-1], keras.Model):
                    return True
            elif isinstance(model, keras.Model):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False
        return False

    def load_model_from_artifact(self, artifact_path):
        from keras.models import load_model

        self._model = load_model(artifact_path)
        self._model._make_predict_function()
        return self._model

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(KerasPredictor, self).predict(data, model, **kwargs)
        predictions = model.predict(data)
        if self.positive_class_label is not None and self.negative_class_label is not None:
            if predictions.shape[1] == 1:
                predictions = pd.DataFrame(predictions, columns=[self.positive_class_label])
                predictions[self.negative_class_label] = 1 - predictions[self.positive_class_label]
            else:
                predictions = pd.DataFrame(
                    predictions, columns=[self.negative_class_label, self.positive_class_label]
                )
        else:
            predictions = pd.DataFrame(predictions, columns=[REGRESSION_PRED_COLUMN])

        return predictions
