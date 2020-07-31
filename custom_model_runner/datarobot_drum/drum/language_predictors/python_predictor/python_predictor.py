import logging
import sys

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

    def configure(self, params):
        super(PythonPredictor, self).configure(params)

        self._model_adapter = PythonModelAdapter(model_dir=self._custom_model_path)

        sys.path.append(self._custom_model_path)
        self._model_adapter.load_custom_hooks()
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

    def predict(self, df):
        kwargs = {}
        if self._positive_class_label and self._negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label

        predictions = self._model_adapter.predict(data=df, model=self._model, **kwargs)
        return predictions
