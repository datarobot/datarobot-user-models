import logging
import sys

from mlpiper.components.connectable_component import ConnectableComponent
from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonPredictor(ConnectableComponent):
    def __init__(self, engine):
        super(PythonPredictor, self).__init__(engine)
        self.model = None
        self.positive_class_label = None
        self.negative_class_label = None
        self.custom_model_path = None
        self._model_adapter = None

    def configure(self, params):
        super(PythonPredictor, self).configure(params)
        self.custom_model_path = self._params["__custom_model_path__"]
        self.positive_class_label = self._params.get("positiveClassLabel")
        self.negative_class_label = self._params.get("negativeClassLabel")
        self._model_adapter = PythonModelAdapter(model_dir=self.custom_model_path)

        sys.path.append(self.custom_model_path)
        self._model_adapter.load_custom_hooks()
        self.model = self._model_adapter.load_model_from_artifact()
        if self.model is None:
            raise Exception("Failed to load model")

    def _materialize(self, parent_data_objs, user_data):
        df = parent_data_objs[0]

        kwargs = {}
        if self.positive_class_label and self.negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self.positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self.negative_class_label

        predictions = self._model_adapter.predict(data=df, model=self.model, **kwargs)
        return [predictions]
