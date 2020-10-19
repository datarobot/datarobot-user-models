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

        if self.target_type.value in TargetType.CLASSIFICATION.value:
            name_to_label = {
                field.name: field.value
                for field in model.outputFields
                if field.feature == "probability"
            }
            actual_name_to_lower_label = {k: v.lower() for k, v in name_to_label.items()}
            expected_lower_labels = [label.lower() for label in self.class_labels]
            # The PMML file may change the case of labels, so we should validate using lower
            if not all(
                label.lower() in actual_name_to_lower_label.values()
                for label in expected_lower_labels
            ):
                raise DrumCommonException(
                    "Target type '{}' predictions must return the "
                    "probability distribution for all class labels {}. "
                    "Predictions had {} columns".format(
                        self.target_type, self.class_labels, predictions.columns
                    )
                )
            # The output may have multiple probability columns for each label.
            # Assume the first one is the correct one.
            pred_columns = [
                next(
                    name
                    for name, lower_label in actual_name_to_lower_label.items()
                    if lower_label == expected_class
                )
                for expected_class in expected_lower_labels
            ]
            predictions = predictions[pred_columns]
            # Rename the prediction columns with the expected name from the model.
            predictions = predictions.rename(
                columns=lambda col: next(
                    label
                    for label in self.class_labels
                    if label.lower() == actual_name_to_lower_label[col]
                )
            )
        elif self.target_type in [TargetType.REGRESSION, TargetType.ANOMALY]:
            predictions = predictions.rename(
                columns={predictions.columns[0]: REGRESSION_PRED_COLUMN}
            )
        else:
            raise DrumCommonException(
                "Target type '{}' is not supported by '{}' predictor".format(
                    self.target_type.value, self.__class__.__name__
                )
            )

        return predictions
