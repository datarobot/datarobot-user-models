#
#  Copyright 2025 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from argparse import Namespace
from unittest import mock

# Import the function under test
from datarobot_drum.drum.utils.setup import setup_options


def test_setup_options_default(monkeypatch):
    # Mock CMRunnerArgsRegistry and dependencies
    mock_parser = mock.Mock()
    mock_options = Namespace(max_workers=None, runtime_params_file=None, lazy_loading_file=None)
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ):
        opts = setup_options([])
        assert hasattr(opts, "max_workers")
        assert opts.max_workers == 1  # Default to 1


def test_setup_options_with_max_workers(monkeypatch):
    mock_parser = mock.Mock()
    mock_options = Namespace(max_workers="3", runtime_params_file=None, lazy_loading_file=None)
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ):
        opts = setup_options([])
        assert opts.max_workers == 3 or opts.max_workers == "3" or int(opts.max_workers) == 3


def test_setup_options_runtime_parameters(monkeypatch):
    mock_parser = mock.Mock()
    mock_options = Namespace(max_workers=None, runtime_params_file=None, lazy_loading_file=None)
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ), mock.patch(
        "datarobot_drum.RuntimeParameters.has", return_value=True
    ), mock.patch(
        "datarobot_drum.RuntimeParameters.get", return_value=5
    ):
        opts = setup_options([])
        assert opts.max_workers == 5


def test_setup_options_runtime_params_file(monkeypatch):
    mock_parser = mock.Mock()
    mock_options = Namespace(
        max_workers=None, runtime_params_file="params.yaml", code_dir=".", lazy_loading_file=None
    )
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ), mock.patch(
        "datarobot_drum.drum.utils.setup.RuntimeParametersLoader"
    ) as mock_loader:
        instance = mock_loader.return_value
        instance.setup_environment_variables = mock.Mock()
        opts = setup_options([])
        instance.setup_environment_variables.assert_called_once()


def test_setup_options_lazy_loading_file(monkeypatch):
    mock_parser = mock.Mock()
    mock_options = Namespace(
        max_workers=None, runtime_params_file=None, lazy_loading_file="lazy.yaml"
    )
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ), mock.patch(
        "datarobot_drum.drum.lazy_loading.lazy_loading_handler.LazyLoadingHandler.setup_environment_variables_from_values_file"
    ) as mock_lazy:
        opts = setup_options([])
        mock_lazy.assert_called_once_with("lazy.yaml")


def test_setup_options_argcomplete_missing(monkeypatch, capsys):
    mock_parser = mock.Mock()
    mock_options = Namespace(max_workers=None, runtime_params_file=None, lazy_loading_file=None)
    mock_parser.parse_args.return_value = mock_options

    with mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.get_arg_parser",
        return_value=mock_parser,
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.extend_sys_argv_with_env_vars"
    ), mock.patch(
        "datarobot_drum.drum.args_parser.CMRunnerArgsRegistry.verify_options"
    ), mock.patch.dict(
        "sys.modules", {"argcomplete": None}
    ):
        opts = setup_options([])
        captured = capsys.readouterr()
        assert "autocompletion of arguments is not supported" in captured.err
