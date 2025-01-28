"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json
import os
import subprocess
from pathlib import Path

from datarobot_drum.drum.enum import CustomHooks
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor


class VllmPredictor(BaseOpenAiGpuPredictor):
    NAME = "vLLM"
    DEFAULT_MODEL_DIR = "vllm"
    ENGINE_CONFIG_FILE = "engine_config.json"
    HEALTH_ROUTE = "/health"

    def __init__(self):
        super().__init__()

        self.huggingface_token = self.get_optional_parameter("HuggingFaceToken")
        self.model = self.get_optional_parameter("model")
        # Add support for some common additional params for vLLM
        self.max_model_len = self.get_optional_parameter("max_model_len")
        self.gpu_memory_utilization = self.get_optional_parameter("gpu_memory_utilization")
        self.gpu_count = int(os.environ.get("GPU_COUNT", 0))

    @property
    def num_deployment_stages(self):
        return 3 if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL) else 1

    def download_and_serve_model(self):
        """
        Download OSS LLM model via custom hook or make sure runtime params are set correctly
        to allow vLLM to download from HuggingFace Hub.
        """
        self.run_load_model_hook_idempotent()

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

        code_dir = Path(self._code_dir)
        engine_config_file = code_dir / self.ENGINE_CONFIG_FILE
        default_model_dir = code_dir / self.DEFAULT_MODEL_DIR
        # For advanced users, allow them to specify arbitrary CLI options that we haven't exposed
        # via runtime parameters.
        if engine_config_file.is_file():
            config = json.loads(engine_config_file.read_text())
            if "args" in config:
                self.logger.info(f"Loading CLI args from config file: {engine_config_file}...")
                cmd.extend(config["args"])

        # If model was provided via engine config file, use that...
        if "--model" in cmd:
            pass

        # or, if custom hook loaded the model into the expected place we are done
        elif default_model_dir.is_dir() and list(default_model_dir.iterdir()):
            self.logger.info(f"Default model path ({default_model_dir}) appears to be ready")
            cmd.extend(["--model", str(default_model_dir)])

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
                f" placed in the `{default_model_dir}` directory."
            )

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
            self.openai_process.process = p
            for line in p.stdout:
                self.logger.info(line[:-1])
