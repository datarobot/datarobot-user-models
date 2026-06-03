#
#  Copyright 2025 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import argparse
from unittest.mock import patch, MagicMock

import pytest

from datarobot_drum.drum.enum import ArgumentsOptions, DrumServerType
from datarobot_drum.drum.entry_point import run_drum_server


def _make_options(subparser_name):
    options = argparse.Namespace()
    options.subparser_name = subparser_name
    return options


@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_server_mode_defaults_to_gunicorn(setup_options, runtime_params, main, main_gunicorn):
    setup_options.return_value = _make_options(ArgumentsOptions.SERVER)
    runtime_params.has.return_value = False

    run_drum_server()

    main_gunicorn.assert_called_once()
    main.assert_not_called()


@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_server_mode_explicit_gunicorn(setup_options, runtime_params, main, main_gunicorn):
    setup_options.return_value = _make_options(ArgumentsOptions.SERVER)
    runtime_params.has.return_value = True
    runtime_params.get.return_value = DrumServerType.GUNICORN

    run_drum_server()

    main_gunicorn.assert_called_once()
    main.assert_not_called()


@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_server_mode_werkzeug(setup_options, runtime_params, main, main_gunicorn):
    setup_options.return_value = _make_options(ArgumentsOptions.SERVER)
    runtime_params.has.return_value = True
    runtime_params.get.return_value = DrumServerType.WERKZEUG

    run_drum_server()

    main.assert_called_once()
    main_gunicorn.assert_not_called()


@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_server_mode_unsupported_type_falls_back_to_gunicorn(
    setup_options, runtime_params, main, main_gunicorn
):
    setup_options.return_value = _make_options(ArgumentsOptions.SERVER)
    runtime_params.has.return_value = True
    runtime_params.get.return_value = "tornado"

    run_drum_server()

    main_gunicorn.assert_called_once()
    main.assert_not_called()


@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_non_server_mode_uses_main(setup_options, runtime_params, main, main_gunicorn):
    setup_options.return_value = _make_options(ArgumentsOptions.SCORE)

    run_drum_server()

    main.assert_called_once()
    main_gunicorn.assert_not_called()


@patch("datarobot_drum.drum.entry_point.logging")
@patch("datarobot_drum.drum.entry_point.main_gunicorn")
@patch("datarobot_drum.drum.entry_point.main")
@patch("datarobot_drum.drum.entry_point.RuntimeParameters")
@patch("datarobot_drum.drum.entry_point.setup_options")
def test_unsupported_server_type_logs_warning(
    setup_options, runtime_params, main, main_gunicorn, mock_logging
):
    setup_options.return_value = _make_options(ArgumentsOptions.SERVER)
    runtime_params.has.return_value = True
    runtime_params.get.return_value = "tornado"

    run_drum_server()

    mock_logging.warning.assert_called_once()
    warning_msg = mock_logging.warning.call_args[0][0]
    assert "not supported" in warning_msg
