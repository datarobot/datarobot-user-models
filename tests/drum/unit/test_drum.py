#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

#
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import pytest
import yaml

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.drum import CMRunner
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME
from datarobot_drum.drum.runtime import DrumRuntime


def set_sys_argv(cmd_line_args):
    # This is required because the sys.argv is manipulated by the 'CMRunnerArgsRegistry'
    cmd_line_args = cmd_line_args.copy()
    cmd_line_args.insert(0, sys.argv[0])
    sys.argv = cmd_line_args


def get_args_parser_options(cli_command: List[str]):
    set_sys_argv(cli_command)
    arg_parser = CMRunnerArgsRegistry.get_arg_parser()
    CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
    options = arg_parser.parse_args()
    CMRunnerArgsRegistry.verify_options(options)
    return options


@pytest.fixture
def this_dir():
    return str(Path(__file__).absolute().parent)


@pytest.fixture
def model_metadata_file_factory():
    with TemporaryDirectory() as temp_dirname:

        def _inner(input_dict):
            file_path = Path(temp_dirname) / MODEL_CONFIG_FILENAME
            with file_path.open("w") as fp:
                yaml.dump(input_dict, fp)
            return temp_dirname

        yield _inner


@pytest.fixture
def temp_metadata(environment_id, model_metadata_file_factory):
    metadata = {
        "name": "joe",
        "type": "training",
        "targetType": "regression",
        "environmentID": environment_id,
        "validation": {"input": "hello"},
    }
    yield model_metadata_file_factory(metadata)


@pytest.fixture
def fit_args(temp_metadata):
    return [
        "fit",
        "--code-dir",
        temp_metadata,
        "--input",
        __file__,
        "--target",
        "pronounced-tar-ZHAY",
        "--target-type",
        "regression",
    ]


@pytest.fixture
def score_args(this_dir):
    return [
        "score",
        "--code-dir",
        this_dir,
        "--input",
        __file__,
    ]


@pytest.fixture
def server_args(this_dir):
    return ["server", "--code-dir", this_dir, "--address", "https://allthedice.com"]


@pytest.fixture
def runtime_factory():
    with DrumRuntime() as cm_runner_runtime:

        def inner(cli_args):
            options = get_args_parser_options(cli_args)
            cm_runner_runtime.options = options
            cm_runner = CMRunner(cm_runner_runtime)
            return cm_runner

        yield inner


class TestCMRunnerRunTestPredict:
    def test_thing(self, runtime_factory, fit_args):
        runner = runtime_factory(fit_args)
        runner.run_test_predict()
