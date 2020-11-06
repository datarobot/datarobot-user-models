import logging
import sys
import time

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    CLASS_LABELS_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(PythonPredictor, self).__init__()
        self._model_adapter = None
        self._mlops = None

    def configure(self, params):
        super(PythonPredictor, self).configure(params)

        self._model_adapter = PythonModelAdapter(
            model_dir=self._custom_model_path, target_type=self._target_type
        )

        sys.path.append(self._custom_model_path)
        self._model_adapter.load_custom_hooks()
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

    @property
    def supported_payload_formats(self):
        return self._model_adapter.supported_payload_formats

    def predict(self, input_filename):
        kwargs = {}
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self._target_type
        if self._positive_class_label and self._negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label
        if self._class_labels:
            kwargs[CLASS_LABELS_ARG_KEYWORD] = self._class_labels

        start_predict = time.time()
        predictions = self._model_adapter.predict(input_filename, model=self._model, **kwargs)
        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000

        # TODO: call monitor only if we are in structured mode
        self.monitor(input_filename, predictions, execution_time_ms)

        return predictions

    def predict_unstructured(self, data, **kwargs):
        str_or_tuple = self._model_adapter.predict_unstructured(
            model=self._model, data=data, **kwargs
        )
        if isinstance(str_or_tuple, (str, bytes, type(None))):
            ret = str_or_tuple, None
        elif isinstance(str_or_tuple, tuple):
            ret = str_or_tuple
        else:
            raise DrumCommonException(
                "Wrong type returned in unstructured mode: {}".format(type(str_or_tuple))
            )
        return ret
