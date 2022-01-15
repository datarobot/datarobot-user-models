"""
Copyright 2022 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor
from datarobot_drum.drum.enum import extra_deps, PythonArtifacts, SupportedFrameworks


class ONNXPredictor(ArtifactPredictor):
    def __init__(self):
        super(ONNXPredictor, self).__init__(
            SupportedFrameworks.ONNX, PythonArtifacts.ONNX_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            #TODO: Asli: Any better checks?
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
        #TODO:Asli - implement
        return True
        # if not self.is_framework_present():
        #     return False
        #
        # try:
        #     from sklearn.pipeline import Pipeline
        #     from tensorflow import keras as keras_tf
        #
        #     if isinstance(model, Pipeline):
        #         # check the final estimator in the pipeline is Keras
        #         if isinstance(model[-1], keras_tf.Model):
        #             return True
        #     elif isinstance(model, keras_tf.Model):
        #         return True
        # except Exception as e:
        #     self._logger.debug("Exception: {}".format(e))
        #     return False
        # return False

    def load_model_from_artifact(self, artifact_path):
        import onnxruntime as ort

        self._model = ort.InferenceSession(artifact_path)
        return self._model

    def predict(self, data, model, **kwargs):
        super(ONNXPredictor, self).predict(data, model, **kwargs)

        input_name = [input.name for input in self._model.get_inputs()]
        out_name = [output.name for output in self._model.get_outputs()]
        predictions = self._model.run(out_name, {input_name[0]: data.to_numpy(np.float32)})

        #TODO:Asli - check if we can return labels instead of None
        return predictions[0], None
