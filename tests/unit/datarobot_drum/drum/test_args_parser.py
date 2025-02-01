"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List
from unittest.mock import patch

import pytest
from datarobot_drum.drum.args_parser import CMRunnerArgsRegistry
from datarobot_drum.drum.enum import ArgumentOptionsEnvVars, ArgumentsOptions
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars
from datarobot_drum.drum.root_predictors.utils import _exec_shell_cmd


def set_sys_argv(cmd_line_args):
    # This is required because the sys.argv is manipulated by the 'CMRunnerArgsRegistry'
    cmd_line_args = cmd_line_args.copy()
    cmd_line_args.insert(0, sys.argv[0])
    sys.argv = cmd_line_args


def execute_arg_parser(success=True):
    arg_parser = CMRunnerArgsRegistry.get_arg_parser()
    CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
    if success:
        options = arg_parser.parse_args()
        CMRunnerArgsRegistry.verify_options(options)
    else:
        with pytest.raises(SystemExit):
            options = arg_parser.parse_args()
            CMRunnerArgsRegistry.verify_options(options)


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
def fit_args(this_dir):
    return [
        "fit",
        "--code-dir",
        this_dir,
        "--input",
        __file__,
        "--target",
        "pronounced-tar-ZHAY",
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
def perf_args(this_dir):
    return ["perf-test", "-i", "200", "-s", "1000", "--code-dir", this_dir, "--input", __file__]


@pytest.fixture
def validation_args(this_dir):
    return ["validation", "--code-dir", this_dir, "--input", __file__]


@pytest.fixture
def push_args(this_dir):
    return ["push", "--code-dir", this_dir]


class TestDrumHelp:
    @pytest.mark.parametrize(
        "cmd",
        [
            "{}".format(ArgumentsOptions.MAIN_COMMAND),
            "{} --help".format(ArgumentsOptions.MAIN_COMMAND),
        ],
    )
    def test_drum_help(self, cmd):
        _, stdo, _ = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "usage: drum" in str(stdo)

    def test_drum_bad_subparser(self):
        cmd = "{} some_command".format(ArgumentsOptions.MAIN_COMMAND)
        _, _, stde = _exec_shell_cmd(
            cmd,
            "Failed in {} command line! {}".format(ArgumentsOptions.MAIN_COMMAND, cmd),
            assert_if_fail=False,
        )
        assert "argument subparser_name: invalid choice: 'some_command'" in str(stde)


class TestMulticlassLabelsParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_multiclass_labels(label_parser)
        return test_parser

    @pytest.fixture
    def class_labels(self):
        return list("ABC")

    @pytest.fixture
    def unordered_class_labels(self):
        return list("BAC")

    @pytest.fixture
    def class_labels_with_spaces(self):
        return ["Label {}".format(i) for i in range(4)]

    @pytest.mark.parametrize(
        "valid_labels", ["class_labels", "unordered_class_labels", "class_labels_with_spaces"]
    )
    @pytest.mark.parametrize("as_file", [True, False])
    def test_valid_class_labels(self, request, valid_labels, as_file, parser):
        valid_labels = request.getfixturevalue(valid_labels)
        with NamedTemporaryFile() as f:
            if as_file:
                f.write("\n".join(valid_labels).encode("utf-8"))
                f.flush()
                args = "dummy --class-labels-file {}".format(f.name).split()
            else:
                args = ["dummy", "--class-labels", *valid_labels]
            options = parser.parse_args(args)

        assert options.class_labels == valid_labels

    @pytest.mark.parametrize("as_file", [True, False])
    def test_too_few_labels(self, as_file, parser):
        labels = list("A")
        with NamedTemporaryFile() as f:
            if as_file:
                f.write("\n".join(labels).encode("utf-8"))
                f.flush()
                args = "dummy --class-labels-file {}".format(f.name).split()
            else:
                args = ["dummy", "--class-labels", *labels]

            with pytest.raises(argparse.ArgumentTypeError, match="at least 2"):
                parser.parse_args(args)


class TestPosNegLabelsParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_pos_neg_labels(label_parser)
        return test_parser

    @pytest.fixture
    def numeric_class_labels(self):
        return [1.0, 0.0]

    @pytest.fixture
    def bool_class_labels(self):
        return [True, False]

    @pytest.fixture
    def str_class_labels(self):
        return ["Yes", "No"]

    @pytest.mark.parametrize(
        "valid_labels", ["numeric_class_labels", "bool_class_labels", "str_class_labels"]
    )
    def test_valid_class_labels(self, request, valid_labels, parser):
        valid_labels = request.getfixturevalue(valid_labels)
        args = "dummy --positive-class-label {} --negative-class-label {}".format(
            valid_labels[0], valid_labels[1]
        ).split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        actual_labels = [options.positive_class_label, options.negative_class_label]
        assert all(isinstance(label, str) for label in actual_labels)
        assert actual_labels == [str(label) for label in valid_labels]


class TestStrictValidationParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_strict_validation(label_parser)
        return test_parser

    def test_disable_strict_validation(self, parser):
        args = "dummy --disable-strict-validation".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        assert options.disable_strict_validation

    def test_enable_strict_validation(self, parser):
        # Test the strict validation is enabled by default
        args = "dummy".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)

        assert not options.disable_strict_validation


class TestFitMetricsReportParser(object):
    @pytest.fixture
    def parser(self):
        test_parser = argparse.ArgumentParser()
        subparsers = test_parser.add_subparsers(dest="command")
        label_parser = subparsers.add_parser("dummy")
        CMRunnerArgsRegistry._reg_arg_report_fit_predict_metadata(label_parser)
        return test_parser

    def test_enable_metadata_report(self, parser):
        args = "dummy --enable-fit-metadata".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)
        assert options.enable_fit_metadata

    def test_disable_metadata_report(self, parser):
        args = "dummy".split()
        with patch.object(sys, "argv", args):
            options = parser.parse_args(args)
        assert not options.enable_fit_metadata


class TestBooleanArgumentOptions:
    def test_boolean_argument_options_passed_through_env_var(self):
        args = f"{ArgumentsOptions.MAIN_COMMAND} server  --with-error-server --skip-predict".split()
        with patch.object(sys, "argv", args):
            unset_drum_supported_env_vars()
            CMRunnerArgsRegistry.get_arg_parser()
            os.environ[ArgumentOptionsEnvVars.PRODUCTION] = "True"
            os.environ[ArgumentOptionsEnvVars.MONITOR] = "False"
            os.environ[ArgumentOptionsEnvVars.WITH_ERROR_SERVER] = "1"
            os.environ[ArgumentOptionsEnvVars.SKIP_PREDICT] = "False"
            os.environ[ArgumentOptionsEnvVars.ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS] = "True"
            CMRunnerArgsRegistry.extend_sys_argv_with_env_vars()
            assert ArgumentsOptions.PRODUCTION in sys.argv
            assert ArgumentsOptions.MONITOR not in sys.argv
            assert ArgumentsOptions.WITH_ERROR_SERVER in sys.argv
            assert ArgumentsOptions.SKIP_PREDICT in sys.argv
            assert sys.argv.count(ArgumentsOptions.WITH_ERROR_SERVER) == 1
            assert sys.argv.count(ArgumentsOptions.SKIP_PREDICT) == 1
            assert ArgumentsOptions.ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS in sys.argv
            unset_drum_supported_env_vars()


class TestMonitorArgs:
    @pytest.fixture(scope="session")
    def tmp_dir_with_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_filepath = Path(tmpdirname) / "input.csv"
            open(input_filepath, "wb").write(b"a,b,c\n1,2,3")
            yield str(input_filepath)

    @pytest.fixture
    def minimal_score_cmd_args(self, tmp_dir_with_dataset):
        return ["score", "--code-dir", "/tmp", "--input", tmp_dir_with_dataset]

    @pytest.fixture
    def binary_score_cmd_args(self, minimal_score_cmd_args):
        minimal_score_cmd_args.extend(["--target-type", "binary"])
        return minimal_score_cmd_args

    @pytest.fixture
    def unstructured_score_cmd_args(self, minimal_score_cmd_args):
        minimal_score_cmd_args.extend(["--target-type", "unstructured"])
        return minimal_score_cmd_args

    @pytest.fixture
    def monitor_cmd_args(self):
        return [
            "--monitor",
            "--deployment-id",
            "123",
            "--model-id",
            "456",
            "--monitor-settings",
            "aaa;bbb",
        ]

    @pytest.fixture
    def monitor_embedded_cmd_args_without_monitor_settings(self):
        return [
            "--monitor-embedded",
            "--deployment-id",
            "123",
            "--model-id",
            "456",
            "--webserver",
            "http://aaa.bbb.ccc",
            "--api-token",
            "zzz",
        ]

    @pytest.fixture
    def monitor_env_vars(self):
        os.environ[ArgumentOptionsEnvVars.MONITOR] = "True"
        os.environ["DEPLOYMENT_ID"] = "e123"
        os.environ["MODEL_ID"] = "e456"
        os.environ["MONITOR_SETTINGS"] = "e;aaa;bbb"
        yield
        os.environ.pop(ArgumentOptionsEnvVars.MONITOR)
        os.environ.pop("DEPLOYMENT_ID")
        os.environ.pop("MODEL_ID")
        os.environ.pop("MONITOR_SETTINGS")

    @pytest.fixture
    def monitor_embedded_env_vars(self):
        os.environ[ArgumentOptionsEnvVars.MONITOR_EMBEDDED] = "true"
        os.environ["DEPLOYMENT_ID"] = "e123"
        os.environ["MODEL_ID"] = "e456"
        os.environ["EXTERNAL_WEB_SERVER_URL"] = "e-http://aaa.bbb.ccc"
        os.environ["API_TOKEN"] = "e-zzz"
        yield
        os.environ.pop(ArgumentOptionsEnvVars.MONITOR_EMBEDDED)
        os.environ.pop("DEPLOYMENT_ID")
        os.environ.pop("MODEL_ID")
        os.environ.pop("EXTERNAL_WEB_SERVER_URL")
        os.environ.pop("API_TOKEN")

    def test_binary_monitor_from_cmd_line_args_success(
        self, binary_score_cmd_args, monitor_cmd_args
    ):
        binary_score_cmd_args.extend(monitor_cmd_args)
        set_sys_argv(binary_score_cmd_args)
        execute_arg_parser()

    @pytest.mark.usefixtures("monitor_env_vars")
    def test_binary_monitor_from_env_vars_success(self, binary_score_cmd_args):
        set_sys_argv(binary_score_cmd_args)
        execute_arg_parser()

    @pytest.mark.usefixtures("monitor_env_vars")
    def test_binary_monitor_from_cmd_line_and_env_vars_success(
        self, binary_score_cmd_args, monitor_cmd_args
    ):
        # pop the last 2 elements in order to take them from the environment
        monitor_cmd_args = monitor_cmd_args[0:-2]
        binary_score_cmd_args.extend(monitor_cmd_args)
        set_sys_argv(binary_score_cmd_args)
        execute_arg_parser()

    def test_binary_monitor_embedded_cmd_args_failure(
        self, binary_score_cmd_args, monitor_embedded_cmd_args_without_monitor_settings
    ):
        binary_score_cmd_args.extend(monitor_embedded_cmd_args_without_monitor_settings)
        set_sys_argv(binary_score_cmd_args)
        execute_arg_parser(success=False)

    @pytest.mark.usefixtures("monitor_embedded_env_vars")
    def test_binary_monitor_embedded_from_env_vars_failure(self, binary_score_cmd_args):
        set_sys_argv(binary_score_cmd_args)
        execute_arg_parser(success=False)

    def test_binary_mutually_exclusive_args_failure(
        self,
        binary_score_cmd_args,
        monitor_cmd_args,
        monitor_embedded_cmd_args_without_monitor_settings,
    ):
        args = binary_score_cmd_args
        args.extend(monitor_cmd_args)
        args.extend(monitor_embedded_cmd_args_without_monitor_settings)
        set_sys_argv(args)
        execute_arg_parser(success=False)

    @pytest.mark.parametrize(
        "monitor_settings",
        [(), ("--monitor-settings", "aaa;bbb")],
    )
    def test_unstructured_monitor_embedded_from_cmd_line_args_success(
        self,
        unstructured_score_cmd_args,
        monitor_embedded_cmd_args_without_monitor_settings,
        monitor_settings,
    ):
        unstructured_score_cmd_args.extend(monitor_embedded_cmd_args_without_monitor_settings)
        unstructured_score_cmd_args.extend(monitor_settings)
        set_sys_argv(unstructured_score_cmd_args)
        execute_arg_parser()

    @pytest.mark.parametrize(
        "env_var_key, env_var_value",
        [(None, None), ("MONITOR_SETTINGS", "aaa;bbb")],
    )
    @pytest.mark.usefixtures("monitor_embedded_env_vars")
    def test_unstructured_monitor_from_env_vars_success(
        self, unstructured_score_cmd_args, env_var_key, env_var_value
    ):
        if env_var_key:
            os.environ[env_var_key] = env_var_value
        set_sys_argv(unstructured_score_cmd_args)
        execute_arg_parser()
        if env_var_key:
            os.environ.pop(env_var_key)

    @pytest.mark.usefixtures("monitor_embedded_env_vars")
    def test_unstructured_monitor_from_cmd_line_and_env_vars_success(
        self, unstructured_score_cmd_args, monitor_embedded_cmd_args_without_monitor_settings
    ):
        # pop the last 2 elements in order to take them from the environment
        monitor_embedded_cmd_args_without_monitor_settings = (
            monitor_embedded_cmd_args_without_monitor_settings[0:-2]
        )
        unstructured_score_cmd_args.extend(monitor_embedded_cmd_args_without_monitor_settings)
        set_sys_argv(unstructured_score_cmd_args)
        execute_arg_parser()

    def test_unstructured_monitor_cmd_args_failure(
        self, unstructured_score_cmd_args, monitor_cmd_args
    ):
        unstructured_score_cmd_args.extend(monitor_cmd_args)
        set_sys_argv(unstructured_score_cmd_args)
        execute_arg_parser(success=False)

    @pytest.mark.usefixtures("monitor_env_vars")
    def test_unstructured_monitor_from_env_vars_failure(self, unstructured_score_cmd_args):
        set_sys_argv(unstructured_score_cmd_args)
        execute_arg_parser(success=False)

    def test_unstructured_mutually_exclusive_args_failure(
        self,
        unstructured_score_cmd_args,
        monitor_cmd_args,
        monitor_embedded_cmd_args_without_monitor_settings,
    ):
        args = unstructured_score_cmd_args
        args.extend(monitor_cmd_args)
        args.extend(monitor_embedded_cmd_args_without_monitor_settings)
        set_sys_argv(args)
        execute_arg_parser(success=False)


class TestDrApiAccess:
    @pytest.fixture(
        scope="class",
        params=[
            ["score", "--target-type", "binary", "--code-dir", "/tmp", "--input", __file__],
            [
                "server",
                "--target-type",
                "unstructured",
                "--code-dir",
                "/tmp",
                "--address",
                "127.0.1.1",
            ],
        ],
    )
    def _cli_args(self, request):
        yield request.param

    @contextlib.contextmanager
    def _dr_api_access_env_vars(self, enabled=True, skipped_env_var=None):
        args = {
            ArgumentOptionsEnvVars.ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS: str(enabled),
            "EXTERNAL_WEB_SERVER_URL": "http://aaa.bbb.ccc",
            "API_TOKEN": "zzz123",
        }
        for env_var, value in args.items():
            if env_var != skipped_env_var:
                os.environ[env_var] = value

        yield

        for env_var in args:
            os.environ.pop(env_var, None)

    def test_dr_api_access_from_env_var_success(self, _cli_args):
        with self._dr_api_access_env_vars():
            set_sys_argv(_cli_args)
            execute_arg_parser()

    @pytest.mark.parametrize("missing_mandatory_env_var", ["EXTERNAL_WEB_SERVER_URL", "API_TOKEN"])
    def test_dr_api_access_from_env_var_missing_var(self, _cli_args, missing_mandatory_env_var):
        with self._dr_api_access_env_vars(enabled=True, skipped_env_var=missing_mandatory_env_var):
            set_sys_argv(_cli_args)
            with pytest.raises(SystemExit) as ex:
                execute_arg_parser()

    @pytest.mark.parametrize(
        "missing_mandatory_env_var", [None, "EXTERNAL_WEB_SERVER_URL", "API_TOKEN"]
    )
    def test_dr_api_access_inactive_success(self, _cli_args, missing_mandatory_env_var):
        with self._dr_api_access_env_vars(enabled=False, skipped_env_var=missing_mandatory_env_var):
            set_sys_argv(_cli_args)
            execute_arg_parser()


class TestUserSecretsArgs:
    @pytest.fixture(params=["fit_args", "score_args", "server_args", "push_args"])
    def parametrized_args(self, request):
        yield request.getfixturevalue(request.param)

    def test_no_user_secrets_passed(self, parametrized_args):
        actual = get_args_parser_options(parametrized_args)
        assert actual.user_secrets_mount_path is None
        assert actual.user_secrets_prefix is None

    def test_set_user_secrets(self, parametrized_args, this_dir):
        prefix = "SECRETS"
        parametrized_args.extend(
            ["--user-secrets-mount-path", this_dir, "--user-secrets-prefix", prefix]
        )

        actual = get_args_parser_options(parametrized_args)
        assert actual.user_secrets_mount_path == this_dir
        assert actual.user_secrets_prefix == prefix

    def test_set_user_secrets_from_env_vars(self, parametrized_args, this_dir):
        prefix = "PREFIX_THIS"
        env_vars = {"USER_SECRETS_PREFIX": prefix, "USER_SECRETS_MOUNT_PATH": this_dir}
        with patch.dict(os.environ, env_vars):
            actual = get_args_parser_options(parametrized_args)

        assert actual.user_secrets_mount_path == this_dir
        assert actual.user_secrets_prefix == prefix

    def test_mount_path_can_be_invalid_directory(self, parametrized_args):
        """It is always possible for a give run that one of the other of
        mounted secrets or env vars exists, we don't want to fail on a missing
        mount path."""
        fake_directory = "/not/a/real/directory/"
        parametrized_args.extend(["--user-secrets-mount-path", fake_directory])

        actual = get_args_parser_options(parametrized_args)
        assert actual.user_secrets_mount_path == fake_directory


class TestTritonServerArgs:
    @pytest.fixture(scope="session")
    def tmp_dir_with_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_filepath = Path(tmpdirname) / "input.csv"
            open(input_filepath, "wb").write(b"a,b,c\n1,2,3")
            yield str(input_filepath)

    @pytest.fixture
    def minimal_score_cmd_args(self, tmp_dir_with_dataset):
        return [
            "score",
            "--code-dir",
            "/tmp",
            "--input",
            tmp_dir_with_dataset,
        ]

    @pytest.fixture
    def cleanup_env_vars(self):
        yield
        os.environ.pop("TRITON_HOST", None)
        os.environ.pop("TRITON_HTTP_PORT", None)
        os.environ.pop("TRITON_GRPC_PORT", None)

    @staticmethod
    def parse_triton_server_args(
        triton_host,
        triton_http_port,
        triton_grpc_port,
        minimal_score_cmd_args,
        is_env_variable=False,
    ):
        if is_env_variable:
            if triton_host:
                os.environ["TRITON_HOST"] = triton_host
            if triton_http_port:
                os.environ["TRITON_HTTP_PORT"] = triton_http_port
            if triton_grpc_port:
                os.environ["TRITON_GRPC_PORT"] = triton_grpc_port

        else:
            if triton_host:
                minimal_score_cmd_args.extend(["--triton-host", triton_host])
            if triton_http_port:
                minimal_score_cmd_args.extend(["--triton-http-port", triton_http_port])
            if triton_grpc_port:
                minimal_score_cmd_args.extend(["--triton-grpc-port", triton_grpc_port])

        actual = get_args_parser_options(minimal_score_cmd_args)
        return actual

    @pytest.mark.parametrize(
        "triton_host,expected_host",
        [(None, "http://localhost"), ("http://127.0.0.1", "http://127.0.0.1")],
    )
    @pytest.mark.parametrize(
        "triton_http_port,expected_http_port", [(None, "8000"), ("8888", "8888")]
    )
    @pytest.mark.parametrize(
        "triton_grpc_port,expected_grpc_port", [(None, "8001"), ("9999", "9999")]
    )
    @pytest.mark.parametrize("is_env_variable", [True, False])
    @pytest.mark.usefixtures("cleanup_env_vars")
    def test_read_triton_server_configs_success(
        self,
        triton_grpc_port,
        expected_grpc_port,
        triton_http_port,
        expected_http_port,
        triton_host,
        expected_host,
        is_env_variable,
        minimal_score_cmd_args,
    ):
        actual = self.parse_triton_server_args(
            triton_host, triton_http_port, triton_grpc_port, minimal_score_cmd_args, is_env_variable
        )
        assert actual.triton_host == expected_host
        assert actual.triton_http_port == expected_http_port
        assert actual.triton_grpc_port == expected_grpc_port

    @pytest.mark.parametrize("triton_host", ["localhost", "127.0.0.1"])
    @pytest.mark.parametrize("triton_http_port", ["qwerty", "-1000", "1.001", "0"])
    @pytest.mark.parametrize("triton_grpc_port", ["qwerty", "-1000", "1.001", "0"])
    @pytest.mark.parametrize("is_env_variable", [True, False])
    @pytest.mark.usefixtures("cleanup_env_vars")
    def test_read_triton_server_configs_success(
        self,
        triton_grpc_port,
        triton_http_port,
        triton_host,
        is_env_variable,
        minimal_score_cmd_args,
    ):
        with pytest.raises(SystemExit):
            self.parse_triton_server_args(
                triton_host,
                triton_http_port,
                triton_grpc_port,
                minimal_score_cmd_args,
                is_env_variable,
            )


class TestGetArgOptions:
    @pytest.mark.parametrize(
        "args",
        ["score_args", "server_args", "fit_args", "perf_args", "validation_args", "push_args"],
    )
    def test_get_options(self, request, args):
        args = request.getfixturevalue(args)
        options = get_args_parser_options(args)
        arg_option = CMRunnerArgsRegistry.get_arg_option(options, ArgumentsOptions.CODE_DIR)
        assert options.code_dir
        assert arg_option == options.code_dir

    @pytest.mark.parametrize("args", ["fit_args", "push_args"])
    def test_get_options_invalid(self, request, args):
        args = request.getfixturevalue(args)
        options = get_args_parser_options(args)
        arg_option = CMRunnerArgsRegistry.get_arg_option(
            options, ArgumentsOptions.RUNTIME_PARAMS_FILE
        )
        assert arg_option is None
        assert "runtime_params_file" not in options
        assert "lazy_loading_file" not in options


class TestRuntimeParametersArgs:
    @pytest.mark.parametrize("args", ["score_args", "server_args", "perf_args", "validation_args"])
    def test_runtime_params_valid(self, request, args):
        args = request.getfixturevalue(args)
        args.extend([ArgumentsOptions.RUNTIME_PARAMS_FILE, __file__])
        get_args_parser_options(args)

    @pytest.mark.parametrize("args", ["fit_args", "push_args"])
    def test_runtime_params_invalid(self, request, args):
        args = request.getfixturevalue(args)
        with pytest.raises(SystemExit), NamedTemporaryFile() as f:
            args.extend([ArgumentsOptions.RUNTIME_PARAMS_FILE, f.name])
            get_args_parser_options(args)


class TestMaxWorkersArgs:
    @pytest.mark.parametrize(
        "expected_max_workers, max_workers_args",
        [
            (0, []),
            (7, ["--max-workers", "7"]),
        ],
    )
    def test_max_workers_args_success(self, expected_max_workers, max_workers_args, server_args):
        server_args.extend(max_workers_args)
        actual = get_args_parser_options(server_args)
        assert actual.max_workers == expected_max_workers

    def test_max_workers_args_env_vars(self, server_args):
        max_workers = "123"
        with patch.dict(os.environ, {"MAX_WORKERS": max_workers}):
            actual = get_args_parser_options(server_args)
        assert actual.max_workers == int(max_workers)

    @pytest.mark.parametrize(
        "expected_err_msg, max_workers_args",
        [
            ("--max-workers: must be > 0\n", ["--production", "--max-workers", "0"]),
            ("--max-workers: must be > 0\n", ["--max-workers", "0"]),
            (
                "--max-workers: invalid type_callback value: 'all'\n",
                ["--production", "--max-workers", "all"],
            ),
            ("--max-workers: invalid type_callback value: 'all'\n", ["--max-workers", "all"]),
        ],
    )
    def test_production_args_fail(self, expected_err_msg, max_workers_args, server_args, capsys):
        server_args.extend(max_workers_args)
        with pytest.raises(SystemExit):
            get_args_parser_options(server_args)
        captured = capsys.readouterr()

        assert captured.err.endswith(expected_err_msg)

    def test_production_arg_requires_max_workers(self, server_args, capsys):
        server_args.append("--production")
        with pytest.raises(SystemExit):
            get_args_parser_options(server_args)
        captured = capsys.readouterr()
        assert captured.out.endswith(
            "Production mode requires a non-zero number of workers [--max-workers > 0].\n"
        )


class TestDrPredictTimeSeriesArgs:
    @pytest.fixture(params=["score_args"])
    def base_args(self, request):
        """Base arguments for score commands"""
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize("command", ["server", "fit", "validation", "perf-test", "push"])
    def test_time_series_args_not_available(self, this_dir, command):
        """Test time series args are rejected for non-score commands"""
        base_args = [command, "--code-dir", this_dir]
        
        # Add required args for specific commands
        if command == "server":
            base_args.extend(["--address", "localhost:5000"])
        elif command in ["fit", "validation", "perf-test"]:
            base_args.extend(["--input", __file__])
            
        # Test forecast point
        args = base_args + [
            "--use-datarobot-predict",
            "--forecast-point", "2023-01-01T00:00:00"
        ]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)
            
        # Test prediction dates
        args = base_args + [
            "--use-datarobot-predict",
            "--predictions-start-date", "2023-01-01T00:00:00",
            "--predictions-end-date", "2023-12-31T23:59:59"
        ]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)

    def test_time_series_args_require_dr_predict(self, base_args):
        """Test that time series args require --use-datarobot-predict"""
        # Test with forecast point
        args = base_args + ["--forecast-point", "2023-01-01T00:00:00"]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)

        # Test with prediction dates
        args = base_args + [
            "--predictions-start-date", "2023-01-01T00:00:00",
            "--predictions-end-date", "2023-12-31T23:59:59"
        ]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)

    def test_time_series_score_command_success(self, base_args):
        """Test time series args work with score command"""
        # Test forecast point works
        args = base_args + [
            "--use-datarobot-predict",
            "--forecast-point", "2023-01-01T00:00:00"
        ]
        options = get_args_parser_options(args)
        assert options.use_datarobot_predict
        assert options.forecast_point == "2023-01-01T00:00:00"
        
        # Test prediction dates work 
        args = base_args + [
            "--use-datarobot-predict",
            "--predictions-start-date", "2023-01-01T00:00:00",
            "--predictions-end-date", "2023-12-31T23:59:59"
        ]
        options = get_args_parser_options(args)
        assert options.use_datarobot_predict
        assert options.predictions_start_date == "2023-01-01T00:00:00"
        assert options.predictions_end_date == "2023-12-31T23:59:59"

    def test_server_command_rejects_time_series(self, server_args):
        """Test server command explicitly rejects time series args"""
        args = server_args + [
            "--use-datarobot-predict",
            "--forecast-point", "2023-01-01T00:00:00"
        ]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)

        args = server_args + [
            "--use-datarobot-predict", 
            "--predictions-start-date", "2023-01-01T00:00:00",
            "--predictions-end-date", "2023-12-31T23:59:59"
        ]
        with pytest.raises(SystemExit):
            get_args_parser_options(args)

    @pytest.mark.parametrize("time_series_arg", [
        "--forecast-point", 
        "--predictions-start-date",
        "--predictions-end-date"
    ])
    def test_use_dr_predict_required(self, base_args, time_series_arg):
        """Test that time series args require --use-datarobot-predict"""
        # Test without use_datarobot_predict
        base_args.extend([time_series_arg, "2023-01-01T00:00:00"])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)
            
        # Add use_datarobot_predict and verify it works
        base_args.extend(["--use-datarobot-predict"])
        if time_series_arg == "--predictions-start-date":
            base_args.extend(["--predictions-end-date", "2023-12-31T23:59:59"])
        elif time_series_arg == "--predictions-end-date":
            base_args.extend(["--predictions-start-date", "2023-01-01T00:00:00"])
        options = get_args_parser_options(base_args)
        assert options.use_datarobot_predict

    @pytest.mark.parametrize("timestamp", [
        "2023-01-01T00:00:00",
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:00:00+00:00"
    ])
    def test_valid_forecast_point(self, base_args, timestamp):
        """Test valid forecast point timestamps with use_datarobot_predict"""
        base_args.extend(["--use-datarobot-predict", "--forecast-point", timestamp])
        options = get_args_parser_options(base_args)
        assert options.use_datarobot_predict
        assert options.forecast_point == timestamp

    @pytest.mark.parametrize("start_date,end_date", [
        ("2023-01-01T00:00:00", "2023-12-31T23:59:59"),
        ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
        ("2023-01-01T00:00:00+00:00", "2023-12-31T23:59:59+00:00")
    ])
    def test_valid_prediction_dates(self, base_args, start_date, end_date):
        """Test valid prediction date ranges with use_datarobot_predict"""
        base_args.extend([
            "--use-datarobot-predict",
            "--predictions-start-date", start_date,
            "--predictions-end-date", end_date
        ])
        options = get_args_parser_options(base_args)
        assert options.use_datarobot_predict
        assert options.predictions_start_date == start_date
        assert options.predictions_end_date == end_date

    def test_prediction_dates_require_both(self, base_args):
        """Test that prediction dates must be provided together"""
        # Test with only start date
        base_args.extend([
            "--use-datarobot-predict",
            "--predictions-start-date", "2023-01-01T00:00:00"
        ])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)

        # Test with only end date
        base_args = base_args[:-2]
        base_args.extend([
            "--predictions-end-date", "2023-12-31T23:59:59"
        ])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)

    def test_forecast_point_and_dates_mutually_exclusive(self, base_args):
        """Test that forecast point cannot be used with prediction dates"""
        base_args.extend([
            "--use-datarobot-predict",
            "--forecast-point", "2023-01-01T00:00:00",
            "--predictions-start-date", "2023-01-01T00:00:00",
            "--predictions-end-date", "2023-12-31T23:59:59"
        ])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)

    @pytest.mark.parametrize("invalid_timestamp", [
        "2023-13-01T00:00:00",  # Invalid month
        "2023-01-32T00:00:00",  # Invalid day
        "not-a-timestamp",      # Invalid string
    ])
    def test_invalid_timestamps(self, base_args, invalid_timestamp):
        """Test invalid ISO-8601 timestamps are rejected"""
        # Test with forecast point
        base_args.extend([
            "--use-datarobot-predict",
            "--forecast-point", invalid_timestamp
        ])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)

        # Test with prediction dates
        base_args = base_args[:-2]
        base_args.extend([
            "--predictions-start-date", invalid_timestamp,
            "--predictions-end-date", "2023-12-31T23:59:59"
        ])
        with pytest.raises(SystemExit):
            get_args_parser_options(base_args)

    def test_use_dr_predict_env_var(self, base_args):
        """Test time series args work with USE_DATAROBOT_PREDICT env var"""
        with patch.dict(os.environ, {"USE_DATAROBOT_PREDICT": "true"}):
            base_args.extend(["--forecast-point", "2023-01-01T00:00:00"])
            options = get_args_parser_options(base_args)
            assert options.use_datarobot_predict
            assert options.forecast_point == "2023-01-01T00:00:00"

    def test_invalid_command_with_time_series(self, request):
        """Test time series args not allowed with other commands"""
        for cmd in ['fit_args', 'validation_args', 'perf_args', 'push_args']:
            args = request.getfixturevalue(cmd)
            args.extend([
                "--use-datarobot-predict",
                "--forecast-point", "2023-01-01T00:00:00"
            ])
            with pytest.raises(SystemExit):
                get_args_parser_options(args)

