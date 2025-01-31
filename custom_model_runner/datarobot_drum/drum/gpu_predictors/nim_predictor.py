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
        if not self.gpu_count:
            raise ValueError("Unexpected empty GPU count.")

        self.ngc_token = self.get_optional_parameter("NGC_API_KEY")
        self.model_profile = self.get_optional_parameter("NIM_MODEL_PROFILE")
        self.max_model_len = self.get_optional_parameter("NIM_MAX_MODEL_LEN")
        self.log_level = self.get_optional_parameter("NIM_LOG_LEVEL")

    @property
    def num_deployment_stages(self):
        return 3 if self.python_model_adapter.has_custom_hook(CustomHooks.LOAD_MODEL) else 1

    def download_and_serve_model(self):
        pass  # do nothing