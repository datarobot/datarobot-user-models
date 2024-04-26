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
from pathlib import Path

import requests
from requests import Timeout

from datarobot_drum.drum.enum import CustomHooks
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors import BaseGpuPredictor
from datarobot_drum.resource.drum_server_utils import DrumServerProcess

CODE_DIR = Path(os.environ.get("CODE_DIR", "/opt/code"))


class VllmPredictor(BaseGpuPredictor):
    DEFAULT_MODEL_DIR = CODE_DIR / "vllm"

    def health_check(self) -> typing.Tuple[dict, int]:
        """
        Proxy health checks to vLLM Inference Server
        """
        try:
            health_url = f"http://{self.openai_host}:{self.openai_port}/health"
            response = requests.get(health_url, timeout=5)
            return {"message": response.text}, response.status_code
        except Timeout:
            return {"message": "Timeout waiting for vLLM health route to respond."}, 503

    def download_and_serve_model(self, openai_process: DrumServerProcess):
        """
        Download OSS LLM model via custom hook or make sure runtime params are set correclty
        to allow vLLM to download from HuggingFace Hub.
        """
        if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL):
            try:
                self.python_model_adapter.load_model_from_artifact(skip_predictor_lookup=True)
            except Exception as e:
                raise DrumCommonException(f"An error occurred when loading your artifact: {str(e)}")

        huggingface_token = self.get_optional_parameter("HuggingFaceToken")
        # If custom hook loaded the model into the expected place we are done
        if self.DEFAULT_MODEL_DIR.is_dir() and list(self.DEFAULT_MODEL_DIR.iterdir()):
            self.logger.info(f"Default model path ({self.DEFAULT_MODEL_DIR}) appears to be ready")
            model_or_path = str(self.DEFAULT_MODEL_DIR)

        # Otherwise, we expect a runtime param to have been specified
        elif model := self.get_optional_parameter("model"):
            if os.path.isdir(model) and os.listdir(model):
                self.logger.info(f"`model` runtime parameter points to a valid directory: {model}")
            else:
                if not huggingface_token:
                    raise DrumCommonException(
                        "`HuggingFaceToken` is a required runtime parameter when `model` runtime"
                        " parameter is provided."
                    )
                self.logger.info(f"Will download `{model}` from HuggingFace Hub")
            model_or_path = model
        else:
            raise DrumCommonException(
                "Either the `model` runtime parameter is required or model files must be"
                f" placed in the `{self.DEFAULT_MODEL_DIR}` directory."
            )

        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            self.openai_host,
            "--port",
            self.openai_port,
            "--served-model-name",
            self.model_name,
            "--model",
            model_or_path,
        ]

        # update the path so vllm process can find its libraries
        env = os.environ.copy()
        if huggingface_token:
            env["HF_TOKEN"] = huggingface_token["apiToken"]
        datarobot_venv_path = os.environ.get("VIRTUAL_ENV")
        env["PATH"] = env["PATH"].replace(f"{datarobot_venv_path}/bin:", "")
        with subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        ) as p:
            openai_process.process = p
            for line in p.stdout:
                self.logger.info(line[:-1])

    def terminate(self):
        """
        Shutdown vLLM Inference Server
        """
        if not self.openai_process or not self.openai_process.process:
            self.logger.info("vLLM is not running, skipping shutdown...")
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
