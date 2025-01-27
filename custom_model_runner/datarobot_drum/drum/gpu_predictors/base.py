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
import signal
import sys
import time
import typing
from pathlib import Path
from subprocess import Popen
from threading import Event
from threading import Thread

import numpy as np
import requests
from requests import ConnectionError, Timeout
from requests import codes as http_codes

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import RawPredictResponse
from datarobot_drum.drum.common import SupportedPayloadFormats
from datarobot_drum.drum.enum import CUSTOM_FILE_NAME
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.enum import CustomHooks
from datarobot_drum.drum.enum import PayloadFormat
from datarobot_drum.drum.enum import StructuredDtoKeys
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors import MLOpsStatusReporter
from datarobot_drum.drum.language_predictors.base_language_predictor import (
    BaseLanguagePredictor,
)
from datarobot_drum.drum.server import HTTP_513_DRUM_PIPELINE_ERROR
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerProcess


class ChatRoles:
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class BaseOpenAiGpuPredictor(BaseLanguagePredictor):
    NAME = "Generic OpenAI API"
    DEFAULT_MODEL_NAME = "datarobot-deployed-llm"
    MAX_RESTARTS = 10
    HEALTH_ROUTE = "/"

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
        self._openai_server_watchdog = None
        self._load_model_hook_successful = None
        self.ai_client = None

        # chat input fields
        self.system_prompt_value = self.get_optional_parameter("system_prompt")
        self.user_prompt_column = self.get_optional_parameter("prompt_column_name", "promptText")

        # completions configuration can be changed with Runtime parameters
        self.max_tokens = int(self.get_optional_parameter("max_tokens", 0)) or None
        self.num_choices_per_completion = int(self.get_optional_parameter("n", 0)) or None
        self.temperature = self.get_optional_parameter("temperature") or None

        # used to load custom model hooks
        self.python_model_adapter = None
        # report deployment status events to DataRobot
        self.verify_ssl = self.get_optional_parameter("verifySSL", True)
        self.status_reporter: MLOpsStatusReporter = None

        self._max_watchdog_restarts = self.get_optional_parameter(
            "max_watchdog_restarts", self.MAX_RESTARTS
        )
        self._max_watchdog_backoff = self.get_optional_parameter("max_watchdog_backoff_sec", 300)

        # Have a check in the ctor to we fail early if optional deps are not installed.
        try:
            import openai  # noqa: F401
        except ImportError:
            raise DrumCommonException("OpenAI Python SDK is not installed")

    def supports_chat(self):
        return True

    def _chat(self, completion_create_params, association_id):
        # Force the incoming model name to to match the expected model name because the
        # name isn't very applicable to BYO LLMs.
        completion_create_params["model"] = self.model_name
        return self.ai_client.chat.completions.create(**completion_create_params)

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

    def configure(self, params):
        from openai import OpenAI

        super().configure(params)
        self.python_model_adapter = PythonModelAdapter(
            model_dir=self._code_dir, target_type=self.target_type
        )
        # download model artifacts with a "load_model" hook
        custom_py_paths = self._get_custom_artifacts()
        if custom_py_paths:
            sys.path.append(self._code_dir)
            self.python_model_adapter.load_custom_hooks()

        self.status_reporter = MLOpsStatusReporter(
            mlops_service_url=self.datarobot_endpoint,
            mlops_api_token=self.datarobot_api_token,
            deployment_id=self.deployment_id,
            verify_ssl=self.verify_ssl,
            total_deployment_stages=self.num_deployment_stages,
        )

        self._openai_server_ready_sentinel = Path(self._code_dir) / ".server_ready"
        self._is_shutting_down = Event()
        self.openai_process = DrumServerProcess()
        self.ai_client = OpenAI(
            base_url=f"http://{self.openai_host}:{self.openai_port}/v1", api_key="fake"
        )
        self.openai_server_thread = Thread(
            target=self.download_and_serve_model, name="OpenAI Server"
        )
        self.openai_server_thread.start()

        self._openai_server_watchdog = Thread(
            target=self.watchdog, daemon=True, name="OpenAI Watchdog"
        )
        self._openai_server_watchdog.start()

    def watchdog(self):
        """
        Used as a watchdog thread that will monitor the openai_server_thread and restart it if it
        crashes.
        """
        # OpenAI server can crash due to user misconfiguration and in this case we don't want the
        # watchdog to keep trying to restart it (as it will just fail again). The current logic
        # waits for the server to successfully start once before enabling the watchdog.
        self.logger.info("Starting OpenAI Server watchdog thread in standby mode...")
        while not self._openai_server_ready_sentinel.exists():
            if self._is_shutting_down.is_set():
                return
            time.sleep(7)
        self.logger.info("OpenAI Server is ready; switching watchdog to active mode...")

        restarts_left = self._max_watchdog_restarts
        sleep_time = 2
        while not self._is_shutting_down.is_set():
            self.openai_server_thread.join(timeout=5)
            if not self.openai_server_thread.is_alive():
                if restarts_left == 0:
                    self.logger.error("OpenAI server thread has crashed too many times, exiting...")
                    sys.exit(1)

                # Since these LLM artifacts are large, it is best for us to just restart the
                # OpenAI server to avoid re-downloading the model. If/when we implement an
                # emptyDir volume to store model data that can persist K8s Container restarts,
                # then we can just crash the whole server and let K8s restart us.
                self.logger.error("OpenAI server thread has crashed, restarting...")
                self.openai_server_thread = Thread(target=self.download_and_serve_model)
                self.openai_server_thread.start()
                time.sleep(sleep_time)
                restarts_left -= 1
                sleep_time = min(self._max_watchdog_backoff, sleep_time**2)

    def run_load_model_hook_idempotent(self):
        """
        Runs the load-model hook if it exists and has not been run yet. This is important for
        cases where the OpenAI server thread crashes and we need to restart it but we can't
        be sure that user written load-model hooks are idempotent.
        """
        if self._load_model_hook_successful:
            self.logger.info("Load-model hook has already run successfully; skipping...")
            return

        if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL):
            try:
                self.status_reporter.report_deployment("Running user provided load-model hook...")
                self.python_model_adapter.load_model_from_artifact(skip_predictor_lookup=True)
                self.status_reporter.report_deployment("Load-model hook completed.")
                self._load_model_hook_successful = True
            except Exception as e:
                raise DrumCommonException(
                    f"An error occurred when loading your artifact(s): {str(e)}"
                )

    def _get_custom_artifacts(self):
        code_dir_abspath = os.path.abspath(self._code_dir)

        custom_py_paths = list(Path(code_dir_abspath).rglob("{}.py".format(CUSTOM_FILE_NAME)))

        if len(custom_py_paths) > 1:
            error_mes = (
                "Multiple custom.py files were identified in the code directories sub directories.\n"
                "The following custom model files were found:\n"
            )
            error_mes += "\n".join([str(path) for path in custom_py_paths])
            self.logger.error(error_mes)
            raise DrumCommonException(error_mes)

        return custom_py_paths

    def liveness_probe(self):
        message, status = self.health_check()
        if status == 200 and not self._openai_server_ready_sentinel.exists():
            # Notify watchdog thread that server has successfully started (multiprocess safe)
            self._openai_server_ready_sentinel.touch(exist_ok=True)
        return message, status

    def readiness_probe(self):
        return self.health_check()

    @staticmethod
    def get_optional_parameter(key: str, default_value=None):
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

        # each prompt row sent as a separate completion request
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
        Proxy health checks to OpenAI Inference Server
        """
        if (
            self.openai_process
            and (p := self.openai_process.process)
            and not self._check_process(p)
        ):
            return {"message": f"{self.NAME} has crashed."}, HTTP_513_DRUM_PIPELINE_ERROR

        try:
            health_url = f"http://{self.openai_host}:{self.openai_port}{self.HEALTH_ROUTE}"
            response = requests.get(health_url, timeout=5)
            return {"message": response.text}, response.status_code
        except Timeout:
            return {
                "message": f"Timeout waiting for {self.NAME} health route to respond."
            }, http_codes.SERVICE_UNAVAILABLE
        except ConnectionError as err:
            return {
                "message": f"{self.NAME} server is not ready: {str(err)}"
            }, http_codes.SERVICE_UNAVAILABLE

    @staticmethod
    def _check_process(process: Popen):
        try:
            os.kill(process.pid, 0)
        except OSError:
            return False
        else:
            return True

    def download_and_serve_model(self):
        raise NotImplementedError

    def terminate(self):
        self._is_shutting_down.set()
        self._openai_server_ready_sentinel.unlink(missing_ok=True)
        if not self.openai_process or not self.openai_process.process:
            self.logger.info("OpenAI server is not running, skipping shutdown...")
            return

        pgid = None
        pid = self.openai_process.process.pid
        try:
            pgid = os.getpgid(pid)
            self.logger.info("Sending signal to ProcessGroup: %s", pgid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self.logger.warning("server at pid=%s is already gone", pid)

        assert self.openai_server_thread is not None
        self.openai_server_thread.join(timeout=10)
        if self.openai_server_thread.is_alive():
            if pgid is not None:
                self.logger.warning("Forcefully killing process group: %s", pgid)
                os.killpg(pgid, signal.SIGKILL)
                self.openai_server_thread.join(timeout=5)
            if self.openai_server_thread.is_alive():
                raise TimeoutError("Server failed to shutdown gracefully in allotted time")
