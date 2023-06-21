"""
Copyright 2023 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pytest
import requests
import werkzeug

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.server import HTTP_422_UNPROCESSABLE_ENTITY

from datarobot_drum.resource.drum_server_utils import DrumServerRun

from datarobot_drum.resource.utils import (
    _exec_shell_cmd,
    _create_custom_model_dir,
)

from requests_toolbelt import MultipartEncoder

from tests.drum.constants import (
    PYTHON_TEXT_GENERATION,
    TEXT_GENERATION
)


def test_text_generation_models_cli(resources, tmp_path, framework_env):
    custom_model_dir = _create_custom_model_dir(
        resources, tmp_path, None, TEXT_GENERATION, PYTHON_TEXT_GENERATION
    )

    input_dataset = resources.datasets(None, TEXT_GENERATION)
