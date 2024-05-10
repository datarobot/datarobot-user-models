#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, PropertyMock, patch

import pandas as pd
import pytest
import yaml
from datarobot_drum.drum.adapters.cli.drum_fit_adapter import DrumFitAdapter
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.drum import (
    CMRunner,
    create_custom_inference_model_folder,
    output_in_code_dir,
)
from datarobot_drum.drum.enum import RunLanguage, MODEL_CONFIG_FILENAME
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from datarobot_drum.drum.utils.structured_input_read_utils import StructuredInputReadUtils
from mlpiper.pipeline.executor import Executor


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
        "--target-type",
        "regression",
        "--language",
        "python",
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

    def test_calls_cm_run_test_class_with_other_options(
        self, runtime_factory, fit_args, mock_cm_run_test_class, output_dir
    ):
        mount_path = "/a/b/c"
        prefix = "PREFIX"
        fit_args.extend(["--user-secrets-mount-path", mount_path, "--user-secrets-prefix", prefix])
        runtime_factory(fit_args).run_test_predict()
        cm_run_test_options = mock_cm_run_test_class.call_args[0][0]

        assert cm_run_test_options.user_secrets_mount_path == mount_path
        assert cm_run_test_options.user_secrets_prefix == prefix


@pytest.fixture
def mock_read_structured_input_file_as_df(target):
    with patch.object(StructuredInputReadUtils, "read_structured_input_file_as_df") as mock_func:
        mock_func.return_value = pd.DataFrame(
            [[1, 2, 3], [1, 2, 3]], columns=[target, target + "a", target + "b"]
        )
        yield mock_func


@pytest.fixture
def mock_check_artifacts_and_get_run_language():
    with patch.object(CMRunner, "_check_artifacts_and_get_run_language") as mock_func:
        mock_func.return_value = RunLanguage.PYTHON
        yield


@pytest.fixture
def mock_sample_data_if_necessary():
    with patch.object(DrumFitAdapter, "sample_data_if_necessary") as mock_func:
        yield mock_func


@pytest.fixture
def mock_model_adapter_fit():
    with patch.object(PythonModelAdapter, "fit") as mock_func:
        yield mock_func


@pytest.fixture
def mock_run_test_predict():
    with patch.object(CMRunner, "run_test_predict") as mock_func:
        yield mock_func


@pytest.mark.usefixtures(
    "mock_read_structured_input_file_as_df",
    "mock_check_artifacts_and_get_run_language",
    "mock_model_adapter_fit",
    "mock_run_test_predict",
)
class TestCMRunnerFit:
    def test_calls_model_adapter_fit_correctly(
        self,
        runtime_factory,
        fit_args,
        mock_model_adapter_fit,
        output_dir,
        mock_read_structured_input_file_as_df,
        target,
    ):
        runtime_factory(fit_args).run()
        raw_data: pd.DataFrame = mock_read_structured_input_file_as_df.return_value
        expected_x = raw_data.drop([target], axis=1)
        expected_y = raw_data[target]
        mock_model_adapter_fit.assert_called_once()
        called_kwargs = mock_model_adapter_fit.call_args[1]
        actual_x = called_kwargs.pop("X")
        actual_y = called_kwargs.pop("y")
        expected_other_kwargs = dict(
            output_dir=output_dir,
            row_weights=None,
            parameters={},
            class_order=None,
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
        )
        assert expected_other_kwargs == called_kwargs
        pd.testing.assert_frame_equal(actual_x, expected_x)
        pd.testing.assert_series_equal(actual_y, expected_y)

    def test_calls_model_adapter_fit_with_secrets(
        self,
        runtime_factory,
        fit_args,
        mock_model_adapter_fit,
    ):
        mount_path = "/path/to/secrets"
        prefix = "super-secret"
        fit_args.extend(["--user-secrets-mount-path", mount_path, "--user-secrets-prefix", prefix])

        runtime_factory(fit_args).run()

        called_kwargs = mock_model_adapter_fit.call_args[1]
        assert called_kwargs["user_secrets_mount_path"] == mount_path
        assert called_kwargs["user_secrets_prefix"] == prefix

    def test_calls_run_test_predict(self, runtime_factory, fit_args, mock_run_test_predict):
        runtime_factory(fit_args).run()
        mock_run_test_predict.assert_called_once_with()

    def test_handles_missing_options(self, runtime_factory, fit_args, mock_model_adapter_fit):
        runtime = runtime_factory(fit_args)
        del runtime.options.user_secrets_mount_path
        del runtime.options.user_secrets_prefix
        runtime.run()

        called_kwargs = mock_model_adapter_fit.call_args[1]
        assert called_kwargs["user_secrets_mount_path"] is None
        assert called_kwargs["user_secrets_prefix"] is None


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
    def test_minimal_server_args(
        self, runtime_factory, server_args, mock_mlpiper_configure, this_dir
    ):
        runner = runtime_factory(server_args)
        runner.run()
        expected = {
            "host": "allthedice.com",
            "port": 1234,
            "show_perf": False,
            "run_language": "python",
            "target_type": "regression",
            "positiveClassLabel": None,
            "negativeClassLabel": None,
            "classLabels": None,
            "__custom_model_path__": this_dir,
            "uwsgi_max_workers": None,
            "monitor": "False",
            "monitor_embedded": "False",
            "model_id": "None",
            "deployment_id": None,
            "monitor_settings": None,
            "external_webserver_url": "None",
            "api_token": "None",
            "single_uwsgi_worker": "False",
            "deployment_config": None,
            "allow_dr_api_access": "False",
            "user_secrets_mount_path": None,
            "user_secrets_prefix": None,
            "gpu_predictor": None,
            "triton_host": "http://localhost",
            "triton_http_port": 8000,
            "triton_grpc_port": 8001,
        }

        mock_mlpiper_configure.assert_called_once_with(expected)

    def test_with_user_secrets_mount_path(
        self, server_args, runtime_factory, mock_mlpiper_configure
    ):
        secrets_mount = "/a/b/c"
        server_args.extend(["--user-secrets-mount-path", secrets_mount])
        runtime_factory(server_args).run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_mount_path"] == secrets_mount

    def test_with_user_secrets_prefix(self, server_args, runtime_factory, mock_mlpiper_configure):
        secrets_prefix = "SOME_PREFIX"
        server_args.extend(["--user-secrets-prefix", secrets_prefix])
        runtime_factory(server_args).run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_prefix"] == secrets_prefix

    def test_handles_missing_options(self, server_args, runtime_factory, mock_mlpiper_configure):
        runtime = runtime_factory(server_args)
        del runtime.options.user_secrets_mount_path
        del runtime.options.user_secrets_prefix
        runtime.run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_mount_path"] is None
        assert params["user_secrets_prefix"] is None


@pytest.fixture
def mock_read_csv(module_under_test):
    with patch(f"{module_under_test}.pd.read_csv") as mock_func:
        yield mock_func


@pytest.mark.usefixtures("mock_mlpiper_configure", "mock_run_pipeline", "mock_read_csv")
class TestCMRunnerScore:
    def test_minimal_score_args(
        self, runtime_factory, score_args, mock_mlpiper_configure, this_dir
    ):
        runner = runtime_factory(score_args)
        runner.run()

        expected = {
            "input_filename": __file__,
            "output_filename": ANY,
            "sparse_column_file": None,
            "positiveClassLabel": None,
            "negativeClassLabel": None,
            "classLabels": None,
            "__custom_model_path__": this_dir,
            "run_language": "python",
            "monitor": "False",
            "monitor_embedded": "False",
            "model_id": "None",
            "deployment_id": None,
            "monitor_settings": None,
            "external_webserver_url": "None",
            "api_token": "None",
            "target_type": "regression",
            "query_params": None,
            "content_type": None,
            "allow_dr_api_access": "False",
            "user_secrets_mount_path": None,
            "user_secrets_prefix": None,
            "gpu_predictor": None,
            "triton_host": "http://localhost",
            "triton_http_port": 8000,
        }
        mock_mlpiper_configure.assert_called_once_with(expected)

    def test_with_user_secrets_mount_path(
        self, score_args, runtime_factory, mock_mlpiper_configure
    ):
        secrets_mount = "/a/b/c"
        score_args.extend(["--user-secrets-mount-path", secrets_mount])
        runtime_factory(score_args).run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_mount_path"] == secrets_mount

    def test_with_user_secrets_prefix(self, score_args, runtime_factory, mock_mlpiper_configure):
        secrets_prefix = "SOME_PREFIX"
        score_args.extend(["--user-secrets-prefix", secrets_prefix])
        runtime_factory(score_args).run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_prefix"] == secrets_prefix

    def test_handles_missing_options(self, score_args, runtime_factory, mock_mlpiper_configure):
        runtime = runtime_factory(score_args)
        del runtime.options.user_secrets_mount_path
        del runtime.options.user_secrets_prefix
        runtime.run()

        mock_mlpiper_configure.assert_called_once()
        params = mock_mlpiper_configure.call_args[0][0]
        assert params["user_secrets_mount_path"] is None
        assert params["user_secrets_prefix"] is None


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
