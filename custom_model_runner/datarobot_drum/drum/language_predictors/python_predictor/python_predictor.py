"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import sys

from datarobot_drum.drum.enum import (
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
            model_dir=self._code_dir, target_type=self._target_type
        )

        sys.path.append(self._code_dir)
        self._model_adapter.load_custom_hooks()
        self._model = self._model_adapter.load_model_from_artifact()
        if self._model is None:
            raise Exception("Failed to load model")

    @property
    def supported_payload_formats(self):
        return self._model_adapter.supported_payload_formats

    def model_info(self):
        model_info = super(PythonPredictor, self).model_info()
        model_info.update(self._model_adapter.model_info())
        return model_info

    def has_read_input_data_hook(self):
        return self._model_adapter.has_read_input_data_hook()

    def _predict(self, **kwargs):
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self._target_type
        if self._positive_class_label is not None and self._negative_class_label is not None:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self._positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self._negative_class_label
        if self._class_labels:
            kwargs[CLASS_LABELS_ARG_KEYWORD] = self._class_labels

        return self._model_adapter.predict(model=self._model, **kwargs)

    def _transform(self, **kwargs):
        return self._model_adapter.transform(model=self._model, **kwargs)

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
