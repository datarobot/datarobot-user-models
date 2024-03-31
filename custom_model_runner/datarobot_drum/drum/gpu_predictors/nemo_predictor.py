"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import atexit
import csv
import io
import json
import logging
import os
import signal
import subprocess
from threading import Thread

import numpy as np
import openai
from openai import OpenAI

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX, PayloadFormat, StructuredDtoKeys
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.language_predictors.base_language_predictor import BaseLanguagePredictor
from datarobot_drum.resource.drum_server_utils import DrumServerProcess, wait_for_server

RUNNING_LANG_MSG = "Running environment: NeMo Inferencing Microservice."
DEFAULT_MODEL_NAME = "generic_llm"


class ChatRoles:
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class NemoPredictor(BaseLanguagePredictor):
    def __init__(self):
        super(NemoPredictor, self).__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

        # chat input fields
        self.system_prompt_value = self.get_optional_parameter("system_prompt")
        self.user_prompt_column = self.get_optional_parameter("prompt_column_name", "promptText")
        self.assistant_response_column = self.get_optional_parameter(
            "assistant_column_name", "assistant"
        )

        # completions configuration can be changed with Runtime parameters
        self.startup_timeout_sec = self.get_optional_parameter("startup_timeout_sec", 60)
        self.max_tokens = self.get_optional_parameter("max_tokens", 512)
        self.use_chat_context = self.get_optional_parameter("chat_context", False)
        self.num_choices_per_completion = self.get_optional_parameter("n", 1)
        self.temperature = self.get_optional_parameter("temperature", 0.01)

        # Nemo server configuration is set in the Drop-in environment
        self.gpu_count = os.environ.get("GPU_COUNT")
        if not self.gpu_count:
            raise ValueError("Unexpected empty GPU count.")
        self.health_port = os.environ.get("HEALTH_PORT", "9997")
        self.openai_port = os.environ.get("OPENAI_PORT", "9999")
        self.nemo_host = os.environ.get("NEMO_HOST", "http://localhost")
        self.nemo_port = os.environ.get("NEMO_PORT", "9998")
        self.model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.nemo_process = None
        self.nim_client = None

    def mlpiper_configure(self, params):
        super(NemoPredictor, self).mlpiper_configure(params)

        # start Nemo server and check health it's health
        self.nemo_process = DrumServerProcess()
        self.nemo_thread = Thread(target=self._run_nemo_server, args=(self.nemo_process,))
        atexit.register(self._shutdown_nemo)
        self.nemo_thread.start()
        self._check_nemo_health()
        self.nim_client = OpenAI(base_url=f"{self.nemo_host}:{self.openai_port}/v1", api_key="fake")

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

        user_prompt = lambda row: {
            "role": ChatRoles.USER,
            "content": self._get(row, self.user_prompt_column),
        }
        assistant_prompt = lambda row: {
            "role": ChatRoles.ASSISTANT,
            "content": self._get(row, self.assistant_response_column),
        }

        # all rows are sent in a single completion request, to preserve a chat context
        if self.use_chat_context:
            messages = [
                prompt(row)
                for row in reader
                #  in chat mode user prompts must alternate with assistant prompts
                for prompt in [user_prompt, assistant_prompt]
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

    def _get(self, row, column_name):
        try:
            return row[column_name]
        except KeyError:
            expected_column_names = [self.user_prompt_column]
            if self.use_chat_context:
                expected_column_names.append(self.assistant_response_column)
            raise DrumCommonException(f"Model expects column names '{expected_column_names}'")

    def _create_completions(self, messages, row_id=0):
        if self.system_prompt_value:
            # only the first chat message can have the system role
            messages.insert(0, {"role": ChatRoles.SYSTEM, "content": self.system_prompt_value})

        try:
            completions = self.nim_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                n=self.num_choices_per_completion,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except openai.BadRequestError:
            self.logger.error("Payload: %s", json.dumps(messages), exc_info=True)
            raise DrumCommonException("Bad payload")

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

    def _run_nemo_server(self, nemo_process: DrumServerProcess):
        cmd = [
            "nemollm_inference_ms",
            "--model",
            self.model_name,
            "--health_port",
            self.health_port,
            "--openai_port",
            self.openai_port,
            "--nemo_port",
            self.nemo_port,
            "--num_gpus",
            self.gpu_count,
            "--log_level",
            "info",
        ]

        # update the path so nemollm process can find its libraries
        env = os.environ.copy()
        datarobot_venv_path = os.environ.get("VIRTUAL_ENV")
        env["PATH"] = env["PATH"].replace(f"{datarobot_venv_path}/bin:", "")
        with subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        ) as p:
            nemo_process.process = p
            for line in p.stdout:
                self.logger.info(line[:-1])

    def _check_nemo_health(self):
        nemo_health_url = f"{self.nemo_host}:{self.health_port}/v1/health/ready"
        try:
            self.logger.info("Checking Nemo readiness...")
            wait_for_server(nemo_health_url, timeout=self.startup_timeout_sec)
        except TimeoutError:
            self.logger.error(
                "Nemo inference server is not ready. Please check the logs for more information."
            )
            try:
                self._shutdown_nemo()
            except TimeoutError as e:
                self.logger.error("Nemo server shutdown failure: %s", e)
            raise

    def _shutdown_nemo(self):
        if not self.nemo_process:
            self.logger.info("Nemo is not running, skipping shutdown...")
            return

        pgid = None
        pid = self.nemo_process.process.pid
        try:
            pgid = os.getpgid(pid)
            self.logger.info("Sending signal to ProcessGroup: %s", pgid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self.logger.warning("server at pid=%s is already gone", pid)

        self.nemo_thread.join(timeout=10)
        if self.nemo_thread.is_alive():
            if pgid is not None:
                self.logger.warning("Forcefully killing process group: %s", pgid)
                os.killpg(pgid, signal.SIGKILL)
                self.nemo_thread.join(timeout=5)
            if self.nemo_thread.is_alive():
                raise TimeoutError("Server failed to shutdown gracefully in allotted time")
