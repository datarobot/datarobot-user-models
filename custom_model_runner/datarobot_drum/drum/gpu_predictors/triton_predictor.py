"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import csv
import io
import logging

import numpy as np
import requests
# todo add openai in requirements of Triton/NIM dropin environments
from openai import OpenAI

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    TritonInferenceServerArtifacts,
    TritonInferenceServerBackends,
    UnstructuredDtoKeys, StructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.drum.utils.drum_utils import DrumUtils
from google.protobuf import text_format
from triton_model_config.model_config_pb2 import ModelConfig

RUNNING_LANG_MSG = "Running environment: Triton Inference Server."
default_system_prompt = (
    "You are a helpful AI assistant. Keep short answers of no more than 2 sentences."
)


class TritonPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(TritonPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.triton_host = None
        self.triton_http_port = None
        self.triton_grpc_port = None
        self.model_config = None
        self.nim_client = None

    def mlpiper_configure(self, params):
        super(TritonPredictor, self).mlpiper_configure(params)
        self.triton_host = params.get("triton_host")
        self.triton_http_port = params.get("triton_http_port")
        self.triton_grpc_port = params.get("triton_grpc_port")
        self.nim_client = OpenAI(base_url=f"{self.triton_host}:{self.triton_http_port}/v1", api_key="fake")

        # read model configuration
        model_config_pbtxt = DrumUtils.find_files_by_extensions(
            self._code_dir, TritonInferenceServerArtifacts.ALL
        )
        if len(model_config_pbtxt) == 0:
            raise DrumCommonException("No model configuration found, add a config.pbtxt")
        elif len(model_config_pbtxt) > 1:
            raise DrumCommonException(
                "Found multiple model configurations. Multi-deployments are not supported yet."
            )

        self.model_config = self.read_model_config(model_config_pbtxt[0])
        # check if model is supported by Triton backend
        if not self.is_supported_triton_server_backend():
            raise DrumCommonException(
                f"Unsupported model platform type: {self.model_config.platform or self.model_config.backend}"
            )

    @staticmethod
    def read_model_config(model_config_pbtxt) -> ModelConfig:
        try:
            model_config = ModelConfig()
            with open(model_config_pbtxt, "r") as f:
                config_text = f.read()
                text_format.Merge(config_text, model_config)

            return model_config
        except Exception as e:
            raise DrumCommonException(
                f"Can't read model configuration: {model_config_pbtxt}"
            ) from e

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
        data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        if isinstance(data, bytes):
            data = data.decode("utf8")

        reader = csv.DictReader(io.StringIO(data))
        results = []
        # TODO: I'm not sure if we are meant to treat each row as a separate request or
        # if it is supposed to represent a whole chat history. How we handle this loop
        # will depend on that; for now I'm assuming separate requests.
        for i, row in enumerate(reader):
            self.logger.debug("Row %d: %s", i, row)
            # TODO: use runtime param to get prompt field name like Buzok does
            user_prompt = row["promptText"]
            system_prompt = row.get("system") or default_system_prompt  # don't send empty system prompt

            model_name = self.model_config.name
            completions = self.nim_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                n=1,
                temperature=0.01,
                max_tokens=512,
            )
            self.logger.debug("results: %s", completions)
            results.append(completions.choices[0].message.content)
        return RawPredictResponse(np.array(results), np.array(["completions"]))

    def predict_unstructured(self, data, **kwargs):
        headers = kwargs.get(UnstructuredDtoKeys.HEADERS)
        model_name = self.model_config.name
        resp = requests.post(
            f"{self.triton_host}:{self.triton_http_port}/v2/models/{model_name}/infer",
            data=data,
            headers=headers,
        )
        return resp.text, None

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")
