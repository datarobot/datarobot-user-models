import glob
import json
import os
import re
from tempfile import NamedTemporaryFile

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import requests

from .constants import TESTS_ROOT_PATH
from .utils import (
    _exec_shell_cmd,
    _cmd_add_class_labels,
    _create_custom_model_dir,
)
from .drum_server_utils import DrumServerRun

from datarobot_drum.drum.common import (
    ArgumentsOptions,
    CUSTOM_FILE_NAME,
    CustomHooks,
    PythonArtifacts,
    RunMode,
)
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.drum.utils import CMRunnerUtils

from .constants import (
    TRAINING,
    INFERENCE,
    XGB,
    KERAS,
    KERAS_JOBLIB,
    SKLEARN,
    SIMPLE,
    PYTORCH,
    PYPMML,
    SKLEARN_ANOMALY,
    RDS,
    CODEGEN,
    MOJO,
    POJO,
    MULTI_ARTIFACT,
    CODEGEN_AND_SKLEARN,
    REGRESSION,
    REGRESSION_INFERENCE,
    BINARY,
    ANOMALY,
    PYTHON,
    NO_CUSTOM,
    PYTHON_ALL_HOOKS,
    PYTHON_LOAD_MODEL,
    R,
    R_ALL_HOOKS,
    R_FIT,
    JAVA,
    PYTHON_XGBOOST_CLASS_LABELS_VALIDATION,
    DOCKER_PYTHON_SKLEARN,
    RESPONSE_PREDICTIONS_KEY,
    WEIGHTS_ARGS,
    WEIGHTS_CSV,
    PYTHON_UNSTRUCTURED,
    R_UNSTRUCTURED,
    UNSTRUCTURED,
    WORDS_COUNT_BASIC,
)


class TestUnstructuredMode:
    @pytest.mark.parametrize(
        "framework, problem, language, docker",
        [
            (UNSTRUCTURED, WORDS_COUNT_BASIC, PYTHON_UNSTRUCTURED, None),
            (UNSTRUCTURED, WORDS_COUNT_BASIC, R_UNSTRUCTURED, None),
        ],
    )
    def test_unstructured_models_batch(
        self,
        resources,
        framework,
        problem,
        language,
        docker,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        input_dataset = resources.datasets(framework, problem)

        output = tmp_path / "output"

        cmd = "{} score --code-dir {} --input {} --output {} --unstructured".format(
            ArgumentsOptions.MAIN_COMMAND, custom_model_dir, input_dataset, output
        )

        if docker:
            cmd += " --docker {} --verbose ".format(docker)

        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
        with open(output) as f:
            out_data = f.read()
            assert "6" in out_data
