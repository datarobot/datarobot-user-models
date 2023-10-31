#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from unittest.mock import patch

import pytest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)


@pytest.fixture
def base_configure_params():
    return {
        "__custom_model_path__": "custom_model_path",
        "monitor": False,
        "target_type": TargetType.REGRESSION.value,
    }


@pytest.fixture
def mock_load_model_from_artifact():
    with patch.object(PythonModelAdapter, "load_model_from_artifact") as mock_func:
        yield mock_func


@pytest.mark.usefixtures("mock_load_model_from_artifact")
class TestMLPiperConfigure:
    @pytest.fixture
    def mount_path_key(self):
        return "user_secrets_mount_path"

    @pytest.fixture
    def prefix_key(self):
        return "user_secrets_prefix"

    def test_no_user_secrets(
        self, base_configure_params, mock_load_model_from_artifact, mount_path_key, prefix_key
    ):
        assert mount_path_key not in base_configure_params
        assert prefix_key not in base_configure_params
        predictor = PythonPredictor()
        predictor.mlpiper_configure(base_configure_params)

        mock_load_model_from_artifact.assert_called_once_with(
            user_secrets_mount_path=None, user_secrets_prefix=None,
        )

    def test_with_user_secrets(
        self, base_configure_params, mock_load_model_from_artifact, mount_path_key, prefix_key
    ):
        mount_path = "/secrets/are/here"
        base_configure_params[mount_path_key] = mount_path
        prefix = "SHHHHHHHH"
        base_configure_params[prefix_key] = prefix
        predictor = PythonPredictor()
        predictor.mlpiper_configure(base_configure_params)

        mock_load_model_from_artifact.assert_called_once_with(
            user_secrets_mount_path=mount_path, user_secrets_prefix=prefix,
        )
