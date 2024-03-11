"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import logging

import requests

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, PayloadFormat, UnstructuredDtoKeys
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

RUNNING_LANG_MSG = "Running environment: Triton Inference Server."


class TritonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(TritonPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.triton_host = None
        self.triton_http_port = None
        self.triton_grpc_port = None

    def mlpiper_configure(self, params):
        super(TritonPredictor, self).mlpiper_configure(params)
        self.triton_host = params.get("triton_host")
        self.triton_http_port = params.get("triton_http_port")
        self.triton_http_port = params.get("triton_grpc_port")

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
        headers = kwargs.get(UnstructuredDtoKeys.HEADERS)
        model_name = "densenet_onnx"
        resp = requests.post(
            f"{self.triton_host}:{self.triton_http_port}/v2/models/{model_name}/infer",
            data=data, headers=headers
        )
        return resp.text, None

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")
