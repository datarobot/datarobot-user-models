"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, PayloadFormat
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

RUNNING_LANG_MSG = "Running environment: Triton Inference Server."


class TritonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(TritonPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

    def mlpiper_configure(self, params):
        super(TritonPredictor, self).mlpiper_configure(params)

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        return formats

    def has_read_input_data_hook(self):
        return False

    def _predict(self, **kwargs) -> RawPredictResponse:
        raise DrumCommonException("Predict structured is not supported")

    def predict_unstructured(self, data, **kwargs):
        # TODO make an actual server call
        return "hello world!", None

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")
