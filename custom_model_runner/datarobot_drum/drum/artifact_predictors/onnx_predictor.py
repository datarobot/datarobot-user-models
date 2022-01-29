"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor
from datarobot_drum.drum.enum import extra_deps, PythonArtifacts, SupportedFrameworks, TargetType
from datarobot_drum.drum.exceptions import DrumCommonException

import numpy as np
import pandas as pd


class ONNXPredictor(ArtifactPredictor):
    def __init__(self):
        super(ONNXPredictor, self).__init__(
            SupportedFrameworks.ONNX, PythonArtifacts.ONNX_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            from onnxruntime import onnxruntime_validation

            return True
        except ImportError:
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.ONNX]

    def can_load_artifact(self, artifact_path):
        if not self.is_artifact_supported(artifact_path):
            return False

        try:
            from onnxruntime import onnxruntime_validation

            return True
        except ImportError:
            return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        try:
            import onnxruntime

            return isinstance(model, onnxruntime.InferenceSession)
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False

    def load_model_from_artifact(self, artifact_path):
        import onnxruntime as ort

        self._model = ort.InferenceSession(artifact_path)
        return self._model

    def predict(self, data, model, **kwargs):
        super(ONNXPredictor, self).predict(data, model, **kwargs)

        input_names = [i.name for i in model.get_inputs()]
        session_result = model.run(None, {input_names[0]: data.to_numpy(np.float32)})

        if len(session_result) == 0:
            raise DrumCommonException("ONNX model should return at least 1 output.")

        if len(session_result) == 1:
            preds = session_result[0]
        else:
            preds = self._handle_multiple_outputs(model, session_result)
        return preds, None

    def _handle_multiple_outputs(self, model, session_result):
        if self.target_type not in [TargetType.BINARY, TargetType.MULTICLASS]:
            return session_result[0]
        else:  # For binary / multiclass target types, ONNX models might have the proba output in subsequent fields
            output_names = [o.name for o in model.get_outputs()]
            for idx, out_name in enumerate(output_names):
                if "prob" in out_name:
                    preds = session_result[idx]
                    # Now check for possibly zipmapped probs, eg. [{label1:prob1, label2:prob2}]
                    if isinstance(preds, list):
                        pd_preds = pd.DataFrame(preds)
                        preds = pd_preds.values
                    return preds
            # If no output with "*prob*" in name found, return the first output
            return session_result[0]
