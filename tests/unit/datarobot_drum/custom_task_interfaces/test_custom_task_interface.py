#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import contextlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from unittest.mock import patch

import pytest

from datarobot_drum.custom_task_interfaces.custom_task_interface import (
    CustomTaskInterface,
    secrets_injection_context,
)


@pytest.fixture
def mounted_secrets_factory():
    with TemporaryDirectory(suffix="-secrets") as dir_name:

        def inner(secrets_dict: Dict[str, dict]):
            top_dir = Path(dir_name)
            for k, v in secrets_dict.items():
                target = top_dir / k
                with target.open("w") as fp:
                    json.dump(v, fp)
            return dir_name

        yield inner


@contextlib.contextmanager
def patch_env(prefix, secrets_dict):
    env_dict = {f"{prefix}_{key}": json.dumps(value) for key, value in secrets_dict.items()}
    with patch.dict(os.environ, env_dict):
        yield


class TestSecretsInjectionContext:
    def test_default_empty_secrets(self):
        interface = CustomTaskInterface()
        assert interface.secrets == {}

    def test_load_secrets_no_file_no_env_vars(self):
        interface = CustomTaskInterface()
        with secrets_injection_context(interface, None, None):
            assert interface.secrets == {}
        assert interface.secrets == {}

    def test_load_secrets_mount_path_does_not_exist(self):
        interface = CustomTaskInterface()
        bad_dir = "/nope/not/a/thing"
        with secrets_injection_context(interface, bad_dir, None):
            assert interface.secrets == {}
        assert interface.secrets == {}

    def test_secrets_with_mounted_secrets(self, mounted_secrets_factory):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        secrets_dir = mounted_secrets_factory(secrets)
        interface = CustomTaskInterface()
        with secrets_injection_context(interface, secrets_dir, None):
            assert interface.secrets == secrets
        assert not interface.secrets

    def test_secrets_with_env_vars(self):
        secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        prefix = "MY_SUPER_PREFIX"

        interface = CustomTaskInterface()
        with patch_env(prefix, secrets):
            with secrets_injection_context(interface, None, prefix):
                assert interface.secrets == secrets
        assert not interface.secrets

    def test_secrets_with_mounted_secrets_supersede_env_secrets(self, mounted_secrets_factory):
        mounted_secrets = {
            "ONE": {"credential_type": "basic", "username": "1", "password": "y"},
            "TWO": {"credential_type": "basic", "username": "2", "password": "z"},
        }
        env_secrets = {
            "TWO": {"credential_type": "basic", "username": "superseded", "password": "superseded"},
            "THREE": {"credential_type": "basic", "username": "3", "password": "A"},
        }
        prefix = "MY_SUPER_PREFIX"
        secrets_dir = mounted_secrets_factory(mounted_secrets)
        interface = CustomTaskInterface()

        expected = mounted_secrets.copy()
        expected["THREE"] = env_secrets["THREE"]
        with patch_env(prefix, env_secrets):
            with secrets_injection_context(interface, secrets_dir, prefix):
                assert interface.secrets == expected
        assert not interface.secrets
