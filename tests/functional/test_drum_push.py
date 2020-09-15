from __future__ import absolute_import

import pytest

from datarobot_drum.drum.common import ArgumentsOptions
from ..drum.utils import _exec_shell_cmd

BASE_MODEL_TEMPLATES_DIR = "model_templates"
BASE_DATASET_DIR = "tests/testdata"


class TestDrumPush(object):
    @pytest.mark.parametrize('model_template', ['/python3_sklearn',
                                                '/python3_xgboost',
                                                '/python3_keras_joblib',
                                                '/python3_pytorch',
                                                '/r_lang'])
    def test_drum_push_training(self, model_template, ):
        cmd = "{} push --code-dir {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            BASE_MODEL_TEMPLATES_DIR + '/training' + model_template,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )

    @pytest.mark.parametrize('model_template', ['/python3_sklearn',
                                                '/python3_xgboost',
                                                '/python3_keras_joblib',
                                                '/python3_pytorch',
                                                '/r_lang'])
    def test_drum_push_inference(self, model_template, ):
        cmd = "{} push --code-dir {} --verbose".format(
            ArgumentsOptions.MAIN_COMMAND,
            BASE_MODEL_TEMPLATES_DIR + '/inference' + model_template,
        )
        _exec_shell_cmd(
            cmd, "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd)
        )