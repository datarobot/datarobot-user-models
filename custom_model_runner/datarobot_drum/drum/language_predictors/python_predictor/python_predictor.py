import logging
import sys
import pprint
import time

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    TargetType,
    CustomHooks,
)
from datarobot_drum.drum.model_adapter import PythonModelAdapter
from datarobot_drum.drum.exceptions import DrumCommonException
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
        if self._target_type == TargetType.UNSTRUCTURED:
            for hook_name in [
                CustomHooks.READ_INPUT_DATA,
                CustomHooks.LOAD_MODEL,
                CustomHooks.SCORE,
            ]:
                if self._model_adapter._custom_hooks[hook_name] is None:
                    raise DrumCommonException(
                        "In '{}' mode hook '{}' must be provided.".format(
                            TargetType.UNSTRUCTURED.value, hook_name
                        )
                    )
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

    def predict(self, input_filename):
        kwargs = {}
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self._target_type
        if self._positive_class_label and self._negative_class_label:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label

        start_predict = time.time()
        predictions = self._model_adapter.predict(input_filename, model=self._model, **kwargs)
        end_predict = time.time()
        execution_time_ms = (end_predict - start_predict) * 1000

        # TODO: call monitor only if we are in structured mode
        self.monitor(input_filename, predictions, execution_time_ms)

        return predictions
