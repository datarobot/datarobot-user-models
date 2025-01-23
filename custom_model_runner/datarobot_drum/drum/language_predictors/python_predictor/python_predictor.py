"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging
import shutil
import sys
import tempfile

import datarobot as dr

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    PythonModelAdapter,
    RawPredictResponse,
)
from datarobot_drum.drum.common import to_bool
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    CLASS_LABELS_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    CustomHooks,
    TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException, DrumSerializationError
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


class PythonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(PythonPredictor, self).__init__()
        self._model_adapter = None
        self._mlops_spool_dir = None

    def configure(self, params):
        target_type = TargetType(params.get("target_type"))
        code_dir = params["__custom_model_path__"]

        self._handle_lazy_loading_files()
        self._model_adapter = PythonModelAdapter(model_dir=code_dir, target_type=target_type)

        sys.path.append(code_dir)
        self._model_adapter.load_custom_hooks()

        super(PythonPredictor, self).configure(params)

        try:
            self._model = self._model_adapter.load_model_from_artifact(
                user_secrets_mount_path=params.get("user_secrets_mount_path"),
                user_secrets_prefix=params.get("user_secrets_prefix"),
            )
        except Exception as e:
            raise DrumSerializationError(f"An error occurred when loading your artifact: {str(e)}")
        if self._model is None:
            raise Exception("Failed to load model")

    def _should_enable_mlops(self):
        return super()._should_enable_mlops() or to_bool(self._params.get("monitor_embedded"))

    def supports_chat(self):
        return self._model_adapter.has_custom_hook(CustomHooks.CHAT)

    @property
    def supported_payload_formats(self):
        return self._model_adapter.supported_payload_formats

    def model_info(self):
        model_info = super(PythonPredictor, self).model_info()
        model_info.update(self._model_adapter.model_info())
        return model_info

    def has_read_input_data_hook(self):
        return self._model_adapter.has_read_input_data_hook()

    def _predict(self, **kwargs) -> RawPredictResponse:
        kwargs[TARGET_TYPE_ARG_KEYWORD] = self.target_type
        if self.positive_class_label is not None and self.negative_class_label is not None:
            kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self.positive_class_label
            kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self.negative_class_label
        if self.class_labels:
            kwargs[CLASS_LABELS_ARG_KEYWORD] = self.class_labels

        return self._model_adapter.predict(model=self._model, **kwargs)

    def _transform(self, **kwargs):
        return self._model_adapter.transform(model=self._model, **kwargs)

    def predict_unstructured(self, data, **kwargs):
        if self._mlops:
            kwargs["mlops"] = self._mlops
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

    def _chat(self, completion_create_params, association_id):
        return self._model_adapter.chat(completion_create_params, self._model, association_id)

    def terminate(self):
        if self._mlops:
            self._mlops.shutdown()
            if self._mlops_spool_dir:
                shutil.rmtree(self._mlops_spool_dir)
