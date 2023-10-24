#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch, PropertyMock, ANY

import pandas as pd

#
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import pytest
import yaml
from mlpiper.pipeline.executor import Executor

from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.drum import (
    CMRunner,
    output_in_code_dir,
    create_custom_inference_model_folder,
)
from datarobot_drum.drum.enum import MODEL_CONFIG_FILENAME, RunLanguage
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
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
def module_under_test():
    return "datarobot_drum.drum.drum"


@pytest.fixture
def this_dir():
    return str(Path(__file__).absolute().parent)


@pytest.fixture
def target():
    return "some-target"


@pytest.fixture
def model_metadata_file_factory():
    with TemporaryDirectory(suffix="code-dir") as temp_dirname:

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
def output_dir():
    with TemporaryDirectory(suffix="output-dir") as dir_name:
        yield dir_name


@pytest.fixture
def fit_args(temp_metadata, target, output_dir):
    return [
        "fit",
        "--code-dir",
        temp_metadata,
        "--input",
        __file__,
        "--target",
        target,
        "--target-type",
        "regression",
        "--output",
        output_dir,
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
    return [
        "server",
        "--code-dir",
        this_dir,
        "--address",
        "allthedice.com:1234",
        "--target-type",
        "regression",
        "--language",
        "python",
    ]


@pytest.fixture
def runtime_factory():
    with DrumRuntime() as cm_runner_runtime:

        def inner(cli_args):
            options = get_args_parser_options(cli_args)
            cm_runner_runtime.options = options
            cm_runner = CMRunner(cm_runner_runtime)
            return cm_runner

        yield inner


@pytest.fixture
def mock_input_df(target):
    with patch.object(CMRunner, "input_df", new_callable=PropertyMock) as mock_prop:
        data = [[1, 2, 3]] * 100
        mock_prop.return_value = pd.DataFrame(data, columns=[target, target + "a", target + "b"])
        yield mock_prop


@pytest.fixture
def mock_get_run_language():
    with patch.object(
        CMRunner, "_get_fit_run_language", return_value=RunLanguage.PYTHON
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_cm_run_test_class(module_under_test):
    with patch(f"{module_under_test}.CMRunTests") as mock_class:
        yield mock_class


@pytest.mark.usefixtures("mock_input_df", "mock_get_run_language", "mock_cm_run_test_class")
class TestCMRunnerRunTestPredict:
    def test_calls_cm_run_test_class_correctly(
        self, runtime_factory, fit_args, mock_cm_run_test_class, output_dir
    ):
        runner = runtime_factory(fit_args)
        original_options = runner.options
        original_input = original_options.input
        target_type = runner.target_type
        schema_validator = runner.schema_validator

        expected_options = deepcopy(original_options)

        runner.run_test_predict()

        expected_options.input = ANY
        expected_options.output = os.devnull
        expected_options.code_dir = output_dir

        mock_cm_run_test_class.assert_called_once_with(
            expected_options, target_type, schema_validator
        )
        actual_options = mock_cm_run_test_class.call_args[0][0]
        assert actual_options.input != original_input


@pytest.fixture
def mock_check_artifacts_and_get_run_language():
    with patch.object(CMRunner, "_check_artifacts_and_get_run_language") as mock_func:
        mock_func.return_value = ""


@pytest.fixture
def mock_mlpiper_configure():
    with patch.object(PythonPredictor, "mlpiper_configure") as mock_func:
        yield mock_func


@pytest.fixture
def mock_run_pipeline():
    with patch.object(Executor, "run_pipeline") as mock_func:
        yield mock_func


@pytest.mark.usefixtures("mock_mlpiper_configure", "mock_run_pipeline")
class TestCMRunnerServer:
    # TODO test various calls to mlpiper configure
    #  port some/none, user_secrets_mount_path some/none, user_secrets_prefix some/none,
    def test_thing(self, runtime_factory, server_args, mock_mlpiper_configure):
        runner = runtime_factory(server_args)
        runner.run()
        print()
        print(mock_mlpiper_configure.call_args)
        mock_mlpiper_configure.assert_called_once_with()


class TestUtilityFunctions:
    def test_output_dir_copy(self):
        with TemporaryDirectory() as tempdir:
            # setup
            file = Path(tempdir, "test.py")
            file.touch()
            Path(tempdir, "__pycache__").mkdir()
            out_dir = Path(tempdir, "out")
            out_dir.mkdir()

            # test
            create_custom_inference_model_folder(tempdir, str(out_dir))
            assert Path(out_dir, "test.py").exists()
            assert not Path(out_dir, "__pycache__").exists()
            assert not Path(out_dir, "out").exists()

    def test_output_in_code_dir(self):
        code_dir = "/test/code/is/here"
        output_other = "/test/not/code"
        output_code_dir = "/test/code/is/here/output"
        assert not output_in_code_dir(code_dir, output_other)
        assert output_in_code_dir(code_dir, output_code_dir)
