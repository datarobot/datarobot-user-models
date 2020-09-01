import logging


from abc import ABC, abstractmethod


from datarobot_drum.drum.common import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class BaseLanguagePredictor(ABC):
    def __init__(self):
        self._model = None
        self._positive_class_label = None
        self._negative_class_label = None
        self._custom_model_path = None
        self._params = None

    def configure(self, params):
        self._custom_model_path = params["__custom_model_path__"]
        self._positive_class_label = params.get("positiveClassLabel")
        self._negative_class_label = params.get("negativeClassLabel")
        self._params = params

    @abstractmethod
    def predict(self, input_filename):
        """ Predict on input_filename """
        pass
