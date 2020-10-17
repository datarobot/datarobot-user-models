import pickle
import numpy as np
import pandas as pd
import sys

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
    TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


class PyTorchPredictor(ArtifactPredictor):
    def __init__(self):
        super(PyTorchPredictor, self).__init__(
            SupportedFrameworks.TORCH, PythonArtifacts.TORCH_EXTENSION
        )

    def is_framework_present(self):
        try:
            import torch
            import torch.nn as nn
            from torch.autograd import Variable

            return True
        except ImportError as e:
            self._logger.debug("Got error in imports: {}".format(e))
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.TORCH]

    def can_load_artifact(self, artifact_path):
        if self.is_artifact_supported(artifact_path) and self.is_framework_present():
            return True
        return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False
        try:
            import torch
            import torch.nn as nn

            if isinstance(model, nn.Module):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False

    def load_model_from_artifact(self, artifact_path):
        self._logger.debug("sys_path: {}".format(sys.path))
        import torch

        model = torch.load(artifact_path)
        model.eval()
        return model

    def predict(self, data, model, **kwargs):
        import torch
        from torch.autograd import Variable

        # checking if positive/negative class labels were provided
        # done in the base class
        super(PyTorchPredictor, self).predict(data, model, **kwargs)
        data = Variable(
            torch.from_numpy(data.values if type(data) != np.ndarray else data).type(
                torch.FloatTensor
            )
        )
        with torch.no_grad():
            predictions = model(data).cpu().data.numpy()
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
