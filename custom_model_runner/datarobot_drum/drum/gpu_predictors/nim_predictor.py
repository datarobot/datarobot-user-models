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

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import (
    RawPredictResponse,
)
from datarobot_drum.drum.enum import (
    CLASS_LABELS_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    CustomHooks,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.gpu_predictors.base import BaseOpenAiGpuPredictor


class NIMPredictor(BaseOpenAiGpuPredictor):
    NAME = "NIM"
    DEFAULT_MODEL_DIR = "model-repo"
    ENGINE_CONFIG_FILE = "engine_config.json"
    LEGACY_START_SERVER_SCRIPT = Path("/opt/nim/start-server.sh")
    START_SERVER_SCRIPT = Path("/opt/nim/start_server.sh")
    HEALTH_ROUTE = "/v1/health/ready"

    def __init__(self):
        super().__init__()

        # Server configuration is set in the Drop-in environment
        self.gpu_count = os.environ.get("GPU_COUNT")
        self.ngc_token = self.get_optional_parameter("NGC_API_KEY")
        self.model_profile = self.get_optional_parameter("NIM_MODEL_PROFILE")
        self.max_model_len = self.get_optional_parameter("NIM_MAX_MODEL_LEN")
        self.log_level = self.get_optional_parameter("NIM_LOG_LEVEL")

    @property
    def num_deployment_stages(self):
        return 3 if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL) else 1

    def download_and_serve_model(self):
        """
        Download OSS LLM model via custom hook or make sure runtime params are set correctly
        to allow NIM to download the model from NGC.
        """
        if not self.gpu_count:
            raise ValueError("Unexpected empty GPU count.")

        self.run_load_model_hook_idempotent()

        cmd = ["/opt/nvidia/nvidia_entrypoint.sh"]
        if self.START_SERVER_SCRIPT.is_file():
            cmd.append(str(self.START_SERVER_SCRIPT))
        elif self.LEGACY_START_SERVER_SCRIPT.is_file():
            cmd.append(str(self.LEGACY_START_SERVER_SCRIPT))
        else:
            raise FileNotFoundError(
                f"Unexpected: neither {self.START_SERVER_SCRIPT} nor "
                f"{self.LEGACY_START_SERVER_SCRIPT} exist in the container."
            )

        # update the path so vllm_nvext (e.g. NIM) process can find its libraries
        env = os.environ.copy()
        datarobot_venv_path = os.environ.get("VIRTUAL_ENV")
        env["PATH"] = env["PATH"].replace(f"{datarobot_venv_path}/bin:", "")

        # vllm_nvext is configured via env vars
        # See https://docs.nvidia.com/nim/large-language-models/latest/configuration.html
        env["NIM_SERVED_MODEL_NAME"] = self.served_model_name
        env["NIM_SERVER_PORT"] = str(self.openai_port)
        if self.model_profile:
            env["NIM_MODEL_PROFILE"] = self.model_profile
        if self.max_model_len:
            env["NIM_MAX_MODEL_LEN"] = str(int(self.max_model_len))
        if self.log_level:
            env["NIM_LOG_LEVEL"] = self.log_level

        # Support https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#air-gap-deployment-local-model-directory-route
        code_dir = Path(self._code_dir)
        default_model_dir = Path(self._code_dir) / self.DEFAULT_MODEL_DIR
        if default_model_dir.is_dir() and list(default_model_dir.iterdir()):
            self.logger.info(f"Detected locally stored model; using {default_model_dir}...")
            env["NIM_MODEL_NAME"] = str(default_model_dir)
        elif self.ngc_token:
            env["NGC_API_KEY"] = self.ngc_token["apiToken"]
        else:
            # User probably did something wrong here but leave it as just a warning because they
            # may be doing something custom via the engine_config.json file.
            self.logger.warning(
                "You must set an `NGC_API_KEY` runtime parameter or download the model artifacts "
                f"from a local source to {default_model_dir} in a custom.py:load_model hook."
            )

        engine_config_file = code_dir / self.ENGINE_CONFIG_FILE
        if engine_config_file.is_file():
            config = json.loads(engine_config_file.read_text())
            if "env" in config:
                self.logger.info(f"Loading env vars from config file: {engine_config_file}...")
                env.update(config["env"])

        self.status_reporter.report_deployment("NIM Server is launching...")
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

    def _predict(self, **kwargs) -> RawPredictResponse:
        if self.python_model_adapter.has_custom_hook(CustomHooks.SCORE):
            # This is adapted from the PythonPredictor class
            kwargs[TARGET_TYPE_ARG_KEYWORD] = self.target_type
            if self.positive_class_label is not None and self.negative_class_label is not None:
                kwargs[POSITIVE_CLASS_LABEL_ARG_KEYWORD] = self.positive_class_label
                kwargs[NEGATIVE_CLASS_LABEL_ARG_KEYWORD] = self.negative_class_label
            if self.class_labels:
                kwargs[CLASS_LABELS_ARG_KEYWORD] = self.class_labels

            # Let hook authors know where they can contact the NIM server
            kwargs["base_url"] = f"http://{self.openai_host}:{self.openai_port}"
            kwargs["openai_client"] = self.ai_client
            return self.python_model_adapter.predict(model=self.served_model_name, **kwargs)
        else:
            return super()._predict(**kwargs)

    def predict_unstructured(self, data, **kwargs):
        if not self.python_model_adapter.has_custom_hook(CustomHooks.SCORE_UNSTRUCTURED):
            raise DrumCommonException("The unstructured target type is not supported")

        # Let hook authors know where they can contact the NIM server
        kwargs["base_url"] = f"http://{self.openai_host}:{self.openai_port}"
        kwargs["openai_client"] = self.ai_client

        # This is adapted from the PythonPredictor class
        str_or_tuple = self.python_model_adapter.predict_unstructured(
            model=self.served_model_name, data=data, **kwargs
        )
        if isinstance(str_or_tuple, (str, bytes, type(None))):
            ret = str_or_tuple, None
        elif isinstance(str_or_tuple, tuple):
            ret = str_or_tuple
        else:
            raise DrumCommonException(
                "Wrong type returned in unstructured mode: {}".format(type(str_or_tuple))
            )
        return ret
