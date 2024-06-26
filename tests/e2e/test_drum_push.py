"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from __future__ import absolute_import

import os
import pytest
import yaml

from datarobot_drum.drum.enum import ArgumentsOptions
from tests.conftest import (
    PYTHON,
    REGRESSION,
    BINARY,
    ANOMALY,
    SKLEARN_ANOMALY,
    SKLEARN_BINARY,
    SKLEARN_REGRESSION,
)
from datarobot_drum.resource.utils import _create_custom_model_dir, _exec_shell_cmd

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DATASET_DIR = "mlpiper/testdata"


def get_push_yaml(env_id, data_path, problem, target_name):
    target_name = "targetName: {targetName}".format(targetName=target_name) if target_name else ""
    return """
            name: drumpush-{problemType}
            type: training
            targetType: {problemType}
            environmentID: {environmentID}
            validation:
                input: {dataPath} 
                {maybeTargetName}
        """.format(
        environmentID=env_id, problemType=problem, dataPath=data_path, maybeTargetName=target_name
    )


class TestDrumPush(object):
    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN_REGRESSION, REGRESSION, PYTHON),
            (SKLEARN_BINARY, BINARY, PYTHON),
            (SKLEARN_ANOMALY, ANOMALY, PYTHON),
        ],
    )
    def test_drum_push_training(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
        get_target_factory,
        sklearn_drop_in_env,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
            is_training=True,
            include_metadata=False,
        )

        env_id, _ = sklearn_drop_in_env
        yaml_string = get_push_yaml(
            env_id, resources.datasets(framework, problem), problem, get_target_factory(problem)
        )
        with open(os.path.join(custom_model_dir, "model-metadata.yaml"), "w") as outfile:
            yaml.dump(yaml.safe_load(yaml_string), outfile, default_flow_style=False)

        cmd = "{} push --code-dir {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
