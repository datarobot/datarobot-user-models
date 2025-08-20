#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import argparse
from unittest.mock import Mock, patch

import pytest
from datarobot_drum.drum.main import main


@pytest.mark.parametrize("workers_param, expected_workers", [(None, 0), (1, 1), (10, 10)])
@patch("datarobot_drum.drum.drum.CMRunner", autospec=True)
@patch("datarobot_drum.drum.utils.setup.RuntimeParameters", autospec=True)
@patch("datarobot_drum.drum.utils.setup.RuntimeParametersLoader", autospec=True)
@patch("datarobot_drum.drum.utils.setup.CMRunnerArgsRegistry", autospec=True)
def test_custom_model_workers(
    args_registry, runtime_params_loader, runtime_params, cm_runner, workers_param, expected_workers
):
    options = argparse.Namespace()
    options.max_workers = 0
    options.code_dir = "dir"

    arg_parser = Mock()
    arg_parser.parse_args.return_value = options
    args_registry.get_arg_parser.return_value = arg_parser

    if workers_param:
        runtime_params.has.return_value = True
        runtime_params.get.return_value = workers_param
    else:
        runtime_params.has.return_value = False

    with patch("datarobot_drum.drum.main.setup_otel") as setup_otel_mock:
        setup_otel_mock.return_value = (None, None, None)
        main()

    runtime_params.has.assert_any_call("CUSTOM_MODEL_WORKERS")
    if workers_param:
        runtime_params.get.assert_any_call("CUSTOM_MODEL_WORKERS")
    assert expected_workers == options.max_workers
