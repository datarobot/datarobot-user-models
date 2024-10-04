"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import csv
import io
import json
import logging
import os
import sys
import typing
from pathlib import Path
from threading import Thread

import numpy as np

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    PythonModelAdapter,
    RawPredictResponse,
)
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import (
    CUSTOM_FILE_NAME,
    LOGGER_NAME_PREFIX,
    REMOTE_ARTIFACT_FILE_EXT,
    PayloadFormat,
    StructuredDtoKeys,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors import MLOpsStatusReporter
from datarobot_drum.drum.language_predictors.base_language_predictor import (
    BaseLanguagePredictor,
)
from datarobot_drum.resource.drum_server_utils import DrumServerProcess


class ChatRoles:
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class BaseOpenAiGpuPredictor(BaseLanguagePredictor):
    DEFAULT_MODEL_NAME = "generic_llm"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

        # used by status reporter
        self.datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT", None)
        self.datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN", None)
        self.deployment_id = os.environ.get("MLOPS_DEPLOYMENT_ID", None)

        # server configuration is set in the Drop-in environment
        self.openai_port = os.environ.get("OPENAI_PORT", "9999")
        self.openai_host = os.environ.get("OPENAI_HOST", "localhost")
        self.openai_process = None
        self.openai_server_thread = None
        self.ai_client = None

        # chat input fields
        self.system_prompt_value = self.get_optional_parameter("system_prompt")
        self.user_prompt_column = self.get_optional_parameter("prompt_column_name", "promptText")
        self.assistant_response_column = self.get_optional_parameter(
            "assistant_column_name", "assistant"
        )

        # completions configuration can be changed with Runtime parameters
        self.max_tokens = int(self.get_optional_parameter("max_tokens", 512))
        self.use_chat_context = self.get_optional_parameter("chat_context", False)
        self.num_choices_per_completion = int(self.get_optional_parameter("n", 1))
        self.temperature = self.get_optional_parameter("temperature", 0.01)

        # used to load custom model hooks
        self.python_model_adapter = None
        # report deployment status events to DataRobot
        self.verify_ssl = self.get_optional_parameter("verifySSL", True)
        self.status_reporter: MLOpsStatusReporter = None

        # Have a check in the ctor to we fail early if optional deps are not installed.
        try:
            import openai  # noqa: F401
        except ImportError:
            raise DrumCommonException("OpenAI Python SDK is not installed")

    def has_read_input_data_hook(self):
        return False

    @property
    def model_name(self):
        return self.DEFAULT_MODEL_NAME

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        return formats

    def mlpiper_configure(self, params):
        from openai import OpenAI

        super().mlpiper_configure(params)
        self.python_model_adapter = PythonModelAdapter(
            model_dir=self._code_dir, target_type=self.target_type
        )
        self.status_reporter = MLOpsStatusReporter(
            mlops_service_url=self.datarobot_endpoint,
            mlops_api_token=self.datarobot_api_token,
            deployment_id=self.deployment_id,
            verify_ssl=self.verify_ssl,
            total_deployment_stages=self.num_deployment_stages,
        )

        # download model artifacts with a "load_model" hook or ".remote" artifact
        custom_py_paths, dot_remote_paths = self._get_custom_artifacts()
        if custom_py_paths:
            sys.path.append(self._code_dir)
            self.python_model_adapter.load_custom_hooks()

        elif dot_remote_paths:
            raise DrumCommonException(
                "The '.remote' artifacts are not supported by the current version of DataRobot User models"
            )

        self.openai_process = DrumServerProcess()
        self.ai_client = OpenAI(
            base_url=f"http://{self.openai_host}:{self.openai_port}/v1", api_key="fake"
        )
        self.openai_server_thread = Thread(
            target=self.download_and_serve_model, args=(self.openai_process,)
        )
        self.openai_server_thread.start()

    def _get_custom_artifacts(self):
        code_dir_abspath = os.path.abspath(self._code_dir)

        custom_py_paths = list(Path(code_dir_abspath).rglob("{}.py".format(CUSTOM_FILE_NAME)))
        remote_artifact_paths = list(Path(code_dir_abspath).rglob(REMOTE_ARTIFACT_FILE_EXT))

        if len(custom_py_paths) + len(remote_artifact_paths) > 1:
            error_mes = (
                "Multiple custom.py/.remote files were identified in the code directories sub directories.\n"
                "The following custom model files were found:\n"
            )
            error_mes += "\n".join(
                [str(path) for path in (custom_py_paths + remote_artifact_paths)]
            )
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        return custom_py_paths, remote_artifact_paths

    def liveness_probe(self):
        return self.health_check()

    def readiness_probe(self):
        return self.health_check()

    @staticmethod
    def get_optional_parameter(key, default_value=None):
        if RuntimeParameters.has(key):
            return RuntimeParameters.get(key)
        return default_value

    def _predict(self, **kwargs) -> RawPredictResponse:
        data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        if isinstance(data, bytes):
            data = data.decode("utf8")

        reader = csv.DictReader(io.StringIO(data))
        results = []

        def user_prompt(row):
            return {
                "role": ChatRoles.USER,
                "content": self._get(row, self.user_prompt_column),
            }

        def assistant_prompt(row):
            return {
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
        from openai import BadRequestError

        if self.system_prompt_value:
            # only the first chat message can have the system role
            messages.insert(0, {"role": ChatRoles.SYSTEM, "content": self.system_prompt_value})

        try:
            completions = self.ai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                n=self.num_choices_per_completion,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except BadRequestError:
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

    @property
    def num_deployment_stages(self):
        raise NotImplementedError

    def health_check(self) -> typing.Tuple[dict, int]:
        """
        Proxy health checks to NeMo Inference Server
        """
        raise NotImplementedError

    def download_and_serve_model(self, openai_process: DrumServerProcess):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError
