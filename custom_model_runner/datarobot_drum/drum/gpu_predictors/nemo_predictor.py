"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import csv
import io
import logging
import os

import numpy as np
from openai import OpenAI

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    StructuredDtoKeys, TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor

RUNNING_LANG_MSG = "Running environment: Nemo Inference Microservices."
DEFAULT_MODEL_NAME = "generic-llm"


class NemoPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(NemoPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.prompt_field = None
        self.triton_host = None
        self.triton_http_port = None
        self.nim_client = None

        self.system_prompt = None
        self.use_chat_context = None
        self.max_tokens = None
        self.count_of_completions_to_generate = None
        self.temperature = None

    def mlpiper_configure(self, params):
        super(NemoPredictor, self).mlpiper_configure(params)

        # TODO move up to the BaseLanguagePredictor
        if self.target_type == TargetType.TEXT_GENERATION:
            self.prompt_field = os.environ.get("TARGET_NAME")
            if not self.prompt_field:
                raise ValueError("Unexpected empty target name for text generation!")

        self.triton_host = params.get("triton_host")
        self.triton_http_port = params.get("triton_http_port")
        self.nim_client = OpenAI(base_url=f"{self.triton_host}:{self.triton_http_port}/v1", api_key="fake")

        self.system_prompt = self.get_optional_parameter("system_prompt")
        self.max_tokens = self.get_optional_parameter("max_tokens", 512)
        self.use_chat_context = self.get_optional_parameter("chat_context", False)
        self.count_of_completions_to_generate = self.get_optional_parameter("n", 1)
        self.temperature = self.get_optional_parameter("temperature", 0.01)

    @staticmethod
    def get_optional_parameter(key, default_value=None):
        try:
            return RuntimeParameters.get(key)
        except ValueError:
            return default_value

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
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for i, row in enumerate(reader):
            self.logger.debug("Row %d: %s", i, row)
            user_prompt = {"role": "user", "content": row[self.prompt_field]}

            if self.use_chat_context:  # read history
                messages.append(user_prompt)
            else:  # each row represents a separate prompt
                completions = self._create_completions([user_prompt], i)
                results.extend(completions)

        # include all the chat history in a single completion request
        if self.use_chat_context:
            completions = self._create_completions(messages)
            results.extend(completions)

        return RawPredictResponse(np.array(results), np.array(["row_id", "choice_id", "completions"]))

    def _create_completions(self, messages, row_id=0):
        completions = self.nim_client.chat.completions.create(
            model=DEFAULT_MODEL_NAME,
            messages=messages,
            n=self.count_of_completions_to_generate,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        completion_choices = [
            [row_id, choice_id, choice.message.content]
            for choice_id, choice in enumerate(completions.choices)
        ]

        self.logger.debug("results: %s", completion_choices)
        return completion_choices

    def predict_unstructured(self, data, **kwargs):
        raise DrumCommonException("The unstructured target type is not supported")

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")
