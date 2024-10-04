"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import signal
import subprocess
import typing

import requests
from requests import ConnectionError, Timeout

from datarobot_drum.drum.enum import CustomHooks
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor
from datarobot_drum.drum.server import HTTP_513_DRUM_PIPELINE_ERROR
from datarobot_drum.resource.drum_server_utils import DrumServerProcess


class NIMPredictor(BaseOpenAiGpuPredictor):
    def __init__(self):
        super().__init__()

        # Server configuration is set in the Drop-in environment
        self.gpu_count = os.environ.get("GPU_COUNT")
        if not self.gpu_count:
            raise ValueError("Unexpected empty GPU count.")

        self.ngc_token = self.get_optional_parameter("NGC_API_KEY")
        self.model_profile = self.get_optional_parameter("NIM_MODEL_PROFILE")
        self.max_model_len = self.get_optional_parameter("NIM_MAX_MODEL_LEN")
        self.log_level = self.get_optional_parameter("NIM_LOG_LEVEL")

        self.server_started_event_reported = False

    @property
    def num_deployment_stages(self):
        return 4 if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL) else 2

    def download_and_serve_model(self, openai_process: DrumServerProcess):
        if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL):
            try:
                self.status_reporter.report_deployment("Model artifacts download started...")
                self.python_model_adapter.load_model_from_artifact(skip_predictor_lookup=True)
                self.status_reporter.report_deployment("Model artifacts download completed.")
            except Exception as e:
                raise DrumCommonException(f"An error occurred when loading your artifact: {str(e)}")

        cmd = ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/nim/start-server.sh"]

        # update the path so vllm_nvext process can find its libraries
        env = os.environ.copy()
        datarobot_venv_path = os.environ.get("VIRTUAL_ENV")
        env["PATH"] = env["PATH"].replace(f"{datarobot_venv_path}/bin:", "")
        env["NIM_SERVED_MODEL_NAME"] = self.model_name
        env["NIM_SERVER_PORT"] = str(self.openai_port)
        if self.ngc_token:
            env["NGC_API_KEY"] = self.ngc_token["apiToken"]
        if self.model_profile:
            env["NIM_MODEL_PROFILE"] = self.model_profile
        if self.max_model_len:
            env["NIM_MAX_MODEL_LEN"] = str(int(self.max_model_len))
        if self.log_level:
            env["NIM_LOG_LEVEL"] = self.log_level

        self.status_reporter.report_deployment("NIM Server is launching...")
        with subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        ) as p:
            openai_process.process = p
            for line in p.stdout:
                self.logger.info(line[:-1])

    def health_check(self) -> typing.Tuple[dict, int]:
        """
        Proxy health checks to NIM Server
        """
        if self.openai_server_thread and not self.openai_server_thread.is_alive():
            return {"message": "NIM watchdog has crashed."}, HTTP_513_DRUM_PIPELINE_ERROR

        try:
            health_url = f"http://{self.openai_host}:{self.openai_port}/v1/health/ready"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200 and not self.server_started_event_reported:
                self.status_reporter.report_deployment("NIM Server is ready.")
                self.server_started_event_reported = True

            return {"message": response.text}, response.status_code
        except Timeout:
            return {"message": "Timeout waiting for NIM health route to respond."}, 503
        except ConnectionError as err:
            return {"message": f"NIM server is not ready: {str(err)}"}, 503

    def terminate(self):
        if not self.openai_process or not self.openai_process.process:
            self.logger.info("NIM is not running, skipping shutdown...")
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
