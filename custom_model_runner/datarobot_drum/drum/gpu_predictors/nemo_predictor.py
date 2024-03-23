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
import signal
import subprocess

import numpy as np
from openai import OpenAI

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    PayloadFormat,
    StructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.resource.drum_server_utils import wait_for_server

RUNNING_LANG_MSG = "Running environment: Nemo Inference Microservices."
DEFAULT_MODEL_NAME = "generic_llm"


class ChatRoles:
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class NemoPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(NemoPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
        self.gpu_count = None
        self.nim_client = None
        self.health_port = None
        self.openai_port = None
        self.nemo_host = None
        self.nemo_port = None
        self.nemo_process = None
        self.server_start_timeout = None

        self.model_name = None
        self.prompt_field = None
        self.system_prompt = None
        self.use_chat_context = None
        self.max_tokens = None
        self.num_choices_per_completion = None
        self.temperature = None

    def mlpiper_configure(self, params):
        super(NemoPredictor, self).mlpiper_configure(params)

        self.prompt_field = os.environ.get("TARGET_NAME")
        if not self.prompt_field:
            raise ValueError("Unexpected empty target name for text generation!")

        self.gpu_count = os.environ.get("GPU_COUNT")
        if not self.gpu_count:
            raise ValueError("Unexpected empty GPU count.")

        # Nemo configuration
        self.health_port = os.environ.get("HEALTH_PORT", "9997")
        self.openai_port = os.environ.get("OPENAI_PORT", "9999")
        self.nemo_host = os.environ.get("NEMO_HOST", "http://localhost")
        self.nemo_port = os.environ.get("NEMO_PORT", "9998")
        self.model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.server_start_timeout = os.environ.get("server_timeout_sec", 30)

        # start Nemo server
        self._run_nemo_server()
        self._check_nemo_health()
        self.nim_client = OpenAI(base_url=f"{self.nemo_host}:{self.openai_port}/v1", api_key="fake")

        # completion request configuration
        self.system_prompt = self.get_optional_parameter("system_prompt")
        self.assistant_field = self.get_optional_parameter("assistant_field", "assistant")
        self.max_tokens = self.get_optional_parameter("max_tokens", 512)
        self.use_chat_context = self.get_optional_parameter("chat_context", False)
        self.num_choices_per_completion = self.get_optional_parameter("n", 1)
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

        user_prompt = lambda row: {"role": ChatRoles.USER, "content": row[self.prompt_field]}
        assistant_prompt = lambda row: {"role": ChatRoles.ASSISTANT, "content": row[self.assistant_field]}

        # all rows are sent in a single completion request, to preserve a chat context
        if self.use_chat_context:
            messages = [
                prompt(row)
                for row in reader
                #  in chat mode user prompts must alternate with assistant prompts
                for prompt in {user_prompt, assistant_prompt}
                # skip empty values
                if prompt(row)["content"]
            ]

            completions = self._create_completions(messages)
            results.extend(completions)

        else:  # each prompt row sent as a separate completion request
            for i, row in enumerate(reader):
                self.logger.debug("Row %d: %s", i, row)
                messages = [user_prompt(row)]
                completions = self._create_completions(messages, i)
                results.extend(completions)

        # TODO DRUM has a restriction for text generation targets to return only a single column
        # column_names = ["row_id", "choice_id", "completions"]
        column_names = ["completions"]

        return RawPredictResponse(np.array(results), np.array(column_names))

    def _create_completions(self, messages, row_id=0):
        if self.system_prompt:
            # only the first chat message can have the system role
            messages.insert(0, {"role": ChatRoles.SYSTEM, "content": self.system_prompt})

        completions = self.nim_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=self.num_choices_per_completion,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        completion_choices = [
            # [row_id, choice_id, choice.message.content]
            choice.message.content
            for choice_id, choice in enumerate(completions.choices)
        ]

        self.logger.debug("results: %s", completion_choices)
        return completion_choices

    def predict_unstructured(self, data, **kwargs):
        raise DrumCommonException("The unstructured target type is not supported")

    def _transform(self, **kwargs):
        raise DrumCommonException("Transform feature is not supported")

    def _run_nemo_server(self):
        cmd = [
            "nemollm_inference_ms",
            "--model",
            DEFAULT_MODEL_NAME,
            "--log_level",
            "info",
            "--health_port",
            self.health_port,
            "--openai_port",
            self.openai_port,
            "--nemo_port",
            self.nemo_port,
            "--num_gpus",
            self.gpu_count,
        ]
        self.logger.debug(f"Nemo cmd: {' '.join(cmd)}")
        self.nemo_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

    def _check_nemo_health(self):
        nemo_health_url = f"{self.nemo_host}:{self.health_port}/v1/health/ready"
        try:
            self.logger.info("Checking Nemo readiness...")
            wait_for_server(nemo_health_url, timeout=30)
        except TimeoutError:
            self.logger.error(
                "Nemo inference server is not ready. Please check the logs for more information.")
            try:
                self._shutdown()
            except TimeoutError as e:
                self.logger.error("Nemo server shutdown failure: %s", e)
            raise

    def _shutdown(self):
        self.logger.debug("Shutdown Nemo Inference server")
        if self.nemo_process:
            os.kill(self.nemo_process.pid, signal.SIGTERM)
            os.kill(self.nemo_process.pid, signal.SIGKILL)
