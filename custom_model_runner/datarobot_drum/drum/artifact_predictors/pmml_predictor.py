import pandas as pd

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
)
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


class PMMLPredictor(ArtifactPredictor):
    def __init__(self):
        super(PMMLPredictor, self).__init__(
            SupportedFrameworks.PYPMML, PythonArtifacts.PYPMML_EXTENSION
        )

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.PYPMML]

    def is_framework_present(self):
        try:
            from pypmml import Model

            return True
        except ImportError as e:
            return False

    def can_load_artifact(self, artifact_path):
        return self.is_artifact_supported(artifact_path)

    def load_model_from_artifact(self, artifact_path):
        from pypmml import Model

        model = Model.load(artifact_path)

        return model

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        from pypmml import Model

        if isinstance(model, Model):
            return True
        else:
            return False

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(PMMLPredictor, self).predict(data, model, **kwargs)

        predictions = model.predict(data)

        if self.positive_class_label is not None and self.negative_class_label is not None:
            if predictions.shape[1] == 2:
                predictions = pd.DataFrame(
                    predictions, columns=[self.negative_class_label, self.positive_class_label]
                )
            else:
                predictions = pd.DataFrame(predictions, columns=[self.positive_class_label])
                predictions[self.negative_class_label] = 1 - predictions[self.positive_class_label]
        else:
            predictions = predictions.rename(
                columns={predictions.columns[0]: REGRESSION_PRED_COLUMN}
            )

        return predictions
