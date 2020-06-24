import pickle
import numpy as np
import pandas as pd
import sys

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
)
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

        data = Variable(torch.from_numpy(data.values).type(torch.FloatTensor))
        with torch.no_grad():
            predictions = model(data).cpu().data.numpy()
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
