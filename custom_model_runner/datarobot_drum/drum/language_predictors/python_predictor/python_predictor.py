import logging
import sys
import pprint
import time

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(PythonPredictor, self).__init__()
        self._model_adapter = None
        self._mlops = None

    def configure(self, params):
        super(PythonPredictor, self).configure(params)

        self._model_adapter = PythonModelAdapter(model_dir=self._custom_model_path)

        sys.path.append(self._custom_model_path)
        self._model_adapter.load_custom_hooks()
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

    def predict(self, input_filename):

        kwargs = {}
        if self._positive_class_label and self._negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label

        start_predict = time.time()
        predictions = self._model_adapter.predict(
            input_filename, model=self._model, unstructured_mode=self._unstructured_mode, **kwargs
        )
        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000

        # TODO: call monitor only if we are in structured mode
        self.monitor(input_filename, predictions, execution_time_ms)

        return predictions
