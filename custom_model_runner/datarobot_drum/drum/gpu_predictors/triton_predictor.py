"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import io
import json
import logging

import requests
from requests import Timeout

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    RawPredictResponse,
)
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    TritonInferenceServerBackends,
    UnstructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors.utils import read_model_config
from datarobot_drum.drum.language_predictors.base_language_predictor import (
    BaseLanguagePredictor,
)

RUNNING_LANG_MSG = "Running environment: Triton Inference Server."
INFERENCE_HEADER = "Inference-Header-Content-Length"


class TritonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(TritonPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.triton_host = None
        self.triton_http_port = None
        self.triton_grpc_port = None
        self.model_config = None

    def mlpiper_configure(self, params):
        super(TritonPredictor, self).mlpiper_configure(params)
        self.triton_host = params.get("triton_host")
        self.triton_http_port = params.get("triton_http_port")
        self.triton_grpc_port = params.get("triton_grpc_port")

        self.model_config = read_model_config(self._code_dir)
        # check if model is supported by Triton backend
        if not self.is_supported_triton_server_backend():
            raise DrumCommonException(
                f"Unsupported model platform type: {self.model_config.platform or self.model_config.backend}"
            )

    def is_supported_triton_server_backend(self) -> bool:
        if not self.model_config:
            return False

        return any(
            [
                self.model_config.platform in TritonInferenceServerBackends.ALL,
                self.model_config.backend in TritonInferenceServerBackends.ALL,
            ]
        )

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        return formats

    def has_read_input_data_hook(self):
        return False

    def _predict(self, **kwargs) -> RawPredictResponse:
        raise DrumCommonException("Predict structured is not supported")

    def _get_inference_header_size(self, data):
        try:
            first_line = io.BytesIO(data).readline()
            json.loads(first_line)  # read first line to check it's a valid json string
            return len(first_line)
        except Exception:
            self.logger.warning("Can't calculate the inference header size", exc_info=True)
            # perhaps inference header is not set into payload and request
            # to Triton still may succeed
            return  # do nothing

    def predict_unstructured(self, data, **kwargs):
        headers = kwargs.get(UnstructuredDtoKeys.HEADERS, {})

        # Predictions API does not forward the headers,
        # while Triton expects to get inference header for the Binary Tensor request. See
        # https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_binary_data.html
        if INFERENCE_HEADER.lower() not in set(k.lower() for k in headers):
            inference_header_size = self._get_inference_header_size(data)
            if inference_header_size:
                headers.update({INFERENCE_HEADER: str(inference_header_size)})

        self.logger.debug(headers)
        model_name = self.model_config.name
        resp = requests.post(
            f"{self.triton_host}:{self.triton_http_port}/v2/models/{model_name}/infer",
            data=data,
            headers=headers,
        )
        return resp.text, None

    def liveness_probe(self):
        return self.health_check()

    def readiness_probe(self):
        return self.health_check()

    def health_check(self):
        # TODO: would be good to respond with 513 status code if Triton server has crashed or
        #   if we can detect some other terminal error relating to loading the model provided.
        try:
            triton_health_url = f"{self.triton_host}:{self.triton_http_port}/v2/health/ready"
            response = requests.get(triton_health_url, timeout=5)
            return {"message": response.text}, response.status_code
        except Timeout:
            return {"message": "Timeout waiting for Triton health route to respond."}, 503

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")
