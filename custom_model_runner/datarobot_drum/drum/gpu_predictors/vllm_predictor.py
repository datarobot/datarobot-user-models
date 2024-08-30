"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import signal
import subprocess
import typing
from pathlib import Path

import requests
from requests import ConnectionError, Timeout
from requests import codes as http_codes

from datarobot_drum.drum.enum import CustomHooks
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor
from datarobot_drum.drum.server import HTTP_513_DRUM_PIPELINE_ERROR
from datarobot_drum.resource.drum_server_utils import DrumServerProcess

CODE_DIR = Path(os.environ.get("CODE_DIR", "/opt/code"))


class VllmPredictor(BaseOpenAiGpuPredictor):
    DEFAULT_MODEL_DIR = CODE_DIR / "vllm"
    ENGINE_CONFIG_FILE = CODE_DIR / "engine_config.json"

    def __init__(self):
        super().__init__()
        self.huggingface_token = self.get_optional_parameter("HuggingFaceToken")
        self.model = self.get_optional_parameter("model")
        # Add support for some common additional params for vLLM
        self.max_model_len = self.get_optional_parameter("max_model_len")
        self.gpu_memory_utilization = self.get_optional_parameter("gpu_memory_utilization")
        self.trust_remote_code = self.get_optional_parameter("trust_remote_code")
        self.gpu_count = int(os.environ.get("GPU_COUNT", 0))

    @property
    def num_deployment_stages(self):
        return 3 if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL) else 1

    def health_check(self) -> typing.Tuple[dict, int]:
        """
        Proxy health checks to vLLM Inference Server
        """
        if self.openai_server_thread and not self.openai_server_thread.is_alive():
            return {"message": "vLLM watchdog has crashed."}, HTTP_513_DRUM_PIPELINE_ERROR

        try:
            health_url = f"http://{self.openai_host}:{self.openai_port}/health"
            response = requests.get(health_url, timeout=5)
            return {"message": response.text}, response.status_code
        except Timeout:
            return {
                "message": "Timeout waiting for vLLM health route to respond."
            }, http_codes.SERVICE_UNAVAILABLE
        except ConnectionError as err:
            return {
                "message": f"vLLM server is not ready: {str(err)}"
            }, http_codes.SERVICE_UNAVAILABLE

    def download_and_serve_model(self, openai_process: DrumServerProcess):
        """
        Download OSS LLM model via custom hook or make sure runtime params are set correctly
        to allow vLLM to download from HuggingFace Hub.
        """
        if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL):
            try:
                self.status_reporter.report_deployment("Running user provided load-model hook...")
                self.python_model_adapter.load_model_from_artifact(skip_predictor_lookup=True)
                self.status_reporter.report_deployment("Load-model hook completed.")
            except Exception as e:
                raise DrumCommonException(f"An error occurred when loading your artifact: {str(e)}")

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
        ]

        # For advanced users, allow them to specify arbitrary CLI options that we haven't exposed
        # via runtime parameters.
        if self.ENGINE_CONFIG_FILE.is_file():
            config = json.loads(self.ENGINE_CONFIG_FILE.read_text())
            if "args" in config:
                cmd.extend(config["args"])

        # If model was provided via engine config file, use that...
        if "--model" in cmd:
            pass

        # or, if custom hook loaded the model into the expected place we are done
        elif self.DEFAULT_MODEL_DIR.is_dir() and list(self.DEFAULT_MODEL_DIR.iterdir()):
            self.logger.info(f"Default model path ({self.DEFAULT_MODEL_DIR}) appears to be ready")
            cmd.extend(["--model", str(self.DEFAULT_MODEL_DIR)])

        # otherwise, we expect a runtime param to have been specified
        elif self.model:
            if os.path.isdir(self.model) and os.listdir(self.model):
                self.logger.info(
                    f"`model` runtime parameter points to a valid directory: {self.model}"
                )
            else:
                if not self.huggingface_token:
                    self.logger.warning(
                        "No `HuggingFaceToken` provided, will attempt to download model from"
                        " HuggingFace Hub without authentication."
                    )
                self.logger.info(f"Will download `{self.model}` from HuggingFace Hub")
            cmd.extend(["--model", self.model])
        else:
            raise DrumCommonException(
                "Either the `model` runtime parameter is required or model files must be"
                f" placed in the `{self.DEFAULT_MODEL_DIR}` directory."
            )

        if self.trust_remote_code and "--trust-remote-code" not in cmd:
            cmd.append("--trust-remote-code")
        if self.max_model_len and "--max-model-len" not in cmd:
            cmd.extend(["--max-model-len", str(int(self.max_model_len))])
        if self.gpu_memory_utilization and "--gpu-memory-utilization" not in cmd:
            cmd.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])

        # If the user hasn't already specified the number of GPUs, we will default to using all
        if self.gpu_count > 1 and "--tensor-parallel-size" not in cmd:
            cmd.extend(["--tensor-parallel-size", str(self.gpu_count)])

        env = os.environ.copy()
        if self.huggingface_token:
            env["HF_TOKEN"] = self.huggingface_token["apiToken"]

        # update the path so vllm process can find its libraries
        datarobot_venv_path = os.environ.get("VIRTUAL_ENV")
        env["PATH"] = env["PATH"].replace(f"{datarobot_venv_path}/bin:", "")

        self.status_reporter.report_deployment("vLLM Inference Server is launching...")
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
