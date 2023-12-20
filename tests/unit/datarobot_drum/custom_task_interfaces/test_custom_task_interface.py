#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from unittest.mock import patch, Mock

import pytest

from datarobot_drum.custom_task_interfaces.custom_task_interface import (
    CustomTaskInterface,
    secrets_injection_context,
)
from datarobot_drum.custom_task_interfaces.user_secrets import BasicSecret


@pytest.fixture
def module_under_test():
    return "datarobot_drum.custom_task_interfaces.custom_task_interface"


@pytest.fixture
def secrets():
    return {"ONE": BasicSecret("a", "b")}


@pytest.fixture
def mock_load_secrets(secrets, module_under_test):
    with patch(f"{module_under_test}.load_secrets", return_value=secrets) as mock_func:
        yield mock_func


@pytest.fixture
def mock_patch_outputs_to_scrub_secrets(module_under_test):
    with patch(f"{module_under_test}.patch_outputs_to_scrub_secrets") as mock_func:
        yield mock_func


@pytest.fixture
def mock_reset_outputs_to_allow_secrets(module_under_test):
    with patch(f"{module_under_test}.reset_outputs_to_allow_secrets") as mock_func:
        yield mock_func


@pytest.mark.usefixtures(
    "mock_patch_outputs_to_scrub_secrets",
    "mock_reset_outputs_to_allow_secrets",
    "mock_load_secrets",
)
class TestSecretsInjectionContext:
    def test_default_empty_secrets(self):
        interface = CustomTaskInterface()
        assert interface.secrets == {}

    def test_calls_load_secrets_correctly(self, mock_load_secrets):
        mount_path = Mock()
        env_var_prefix = Mock()
        with secrets_injection_context(CustomTaskInterface(), mount_path, env_var_prefix):
            pass
        mock_load_secrets.assert_called_once_with(
            mount_path=mount_path, env_var_prefix=env_var_prefix
        )

    def test_loads_and_unloads_secrets(self, mock_load_secrets, secrets):
        interface = CustomTaskInterface()
        with secrets_injection_context(interface, Mock(), Mock()):
            assert interface.secrets == secrets
        assert interface.secrets == {}

    def test_patches_outputs(self, mock_patch_outputs_to_scrub_secrets, secrets):
        with secrets_injection_context(CustomTaskInterface(), Mock(), Mock()):
            mock_patch_outputs_to_scrub_secrets.assert_called_once()

        mock_patch_outputs_to_scrub_secrets.assert_called_once()
        # values do not compare to values
        actual = list(mock_patch_outputs_to_scrub_secrets.call_args[0][0])
        expected = list(secrets.values())
        assert actual == expected

    def test_resets_output(self, mock_reset_outputs_to_allow_secrets):
        with secrets_injection_context(CustomTaskInterface(), Mock(), Mock()):
            mock_reset_outputs_to_allow_secrets.assert_not_called()
        mock_reset_outputs_to_allow_secrets.assert_called_once_with()
