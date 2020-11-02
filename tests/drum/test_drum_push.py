from __future__ import absolute_import

import os
import pytest
import yaml

from datarobot_drum.drum.common import ArgumentsOptions
from tests.conftest import PYTHON, SKLEARN, REGRESSION, BINARY
from tests.drum.utils import _create_custom_model_dir, _exec_shell_cmd

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DATASET_DIR = "tests/testdata"


def get_push_yaml(env_id, data_path, problem):
    return """
            name: drumpush-{problemType}
            type: training
            targetType: {problemType}
            environmentID: {environmentID}
            validation:
                input: {dataPath} 
        """.format(
        environmentID=env_id, problemType=problem, dataPath=data_path
    )


class TestDrumPush(object):
    @pytest.mark.parametrize(
        "framework, problem, language",
        [
            (SKLEARN, REGRESSION, PYTHON),
            (SKLEARN, BINARY, PYTHON),
        ],
    )
    def test_drum_push_training(
        self,
        resources,
        framework,
        problem,
        language,
        tmp_path,
    ):
        custom_model_dir = _create_custom_model_dir(
            resources,
            tmp_path,
            framework,
            problem,
            language,
        )

        env_id= '5e8c889607389fe0f466c72d'
        yaml_string = get_push_yaml(env_id, resources.datasets(framework, problem), problem)
        with open(os.path.join(custom_model_dir, "model-metadata.yaml"), "w") as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)

        cmd = "{} push --code-dir {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            custom_model_dir,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )
