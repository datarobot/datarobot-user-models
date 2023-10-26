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
from unittest.mock import Mock, patch

import pytest

from datarobot_drum.custom_task_interfaces.custom_task_interface import secrets_injection_context
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter


class FakeCustomTask:
    def __init__(self):
        self.secrets = None
        self.fit_secrets = Mock()
        self.save_secrets = Mock()
        self.fit_calls = []
        self.save_calls = []

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))
        self.fit_secrets = self.secrets

    def save(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))
        self.save_secrets = self.secrets


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


@pytest.fixture
def secrets_prefix():
    return "PRRRREEEEEEEFFIIIIIXXXX"


@pytest.fixture
def env_secret():
    return {
        "FROM_ENV": {"credential_type": "base", "username": "from-env", "password": "env-password"}
    }


@pytest.fixture
def mounted_secret():
    return {
        "FROM_MOUNTED": {
            "credential_type": "base",
            "username": "from-mounted",
            "password": "mounted-password",
        }
    }


@pytest.fixture
def mounted_secrets_dir(mounted_secret, mounted_secrets_factory):
    yield mounted_secrets_factory(mounted_secret)


@pytest.fixture
def secrets(secrets_prefix, env_secret, mounted_secrets_dir):
    with patch_env(secrets_prefix, env_secret):
        yield


class TestingPythonModelAdapter(PythonModelAdapter):
    def __init__(self, model_dir, target_type):
        super().__init__(model_dir, target_type)
        self._custom_task_class = FakeCustomTask

    @property
    def custom_task_instance(self) -> FakeCustomTask:
        return self._custom_task_class_instance


@pytest.mark.usefixtures("secrets")
class TestFit:
    def test_fit_with_no_secrets(self):
        model_dir = Mock()
        adapter = TestingPythonModelAdapter(model_dir, Mock())
        X = Mock()
        y = Mock()
        output_dir = Mock()
        class_order = Mock()
        row_weights = Mock()
        parameters = Mock()
        adapter.fit(
            X=X,
            y=y,
            output_dir=output_dir,
            class_order=class_order,
            row_weights=row_weights,
            parameters=parameters,
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
        )

        instance = adapter.custom_task_instance
        expected_fit_kwargs = dict(
            X=X,
            y=y,
            output_dir=output_dir,
            class_order=class_order,
            row_weights=row_weights,
            parameters=parameters,
        )
        assert instance.fit_calls == [(tuple(), expected_fit_kwargs)]
        assert instance.save_calls == [((model_dir,), {})]
        assert instance.fit_secrets == {}
        assert instance.save_secrets is None

    def test_fit_with_mounted_secrets(self, mounted_secret, mounted_secrets_dir):
        adapter = TestingPythonModelAdapter(Mock(), Mock())
        adapter.fit(
            X=Mock(),
            y=Mock(),
            output_dir=Mock(),
            class_order=Mock(),
            row_weights=Mock(),
            parameters=Mock(),
            user_secrets_mount_path=mounted_secrets_dir,
            user_secrets_prefix=None,
        )

        instance = adapter.custom_task_instance
        assert instance.fit_secrets == mounted_secret
        assert instance.save_secrets is None

    def test_fit_with_secrets_prefix(self, env_secret, secrets_prefix):
        adapter = TestingPythonModelAdapter(Mock(), Mock())
        adapter.fit(
            X=Mock(),
            y=Mock(),
            output_dir=Mock(),
            class_order=Mock(),
            row_weights=Mock(),
            parameters=Mock(),
            user_secrets_mount_path=None,
            user_secrets_prefix=secrets_prefix,
        )

        instance = adapter.custom_task_instance
        assert instance.fit_secrets == env_secret
        assert instance.save_secrets is None

    def test_got_it_working(self, mounted_secrets_dir, secrets_prefix):
        x = FakeCustomTask()
        print("SECRETS")
        with secrets_injection_context(x, None, None):
            print(f"in None, None: {x.secrets}")
        print(f"out: {x.secrets}")
        with secrets_injection_context(x, mounted_secrets_dir, None):
            print(f"in mount, None: {x.secrets}")
        print(f"out: {x.secrets}")
        with secrets_injection_context(x, None, secrets_prefix):
            print(f"in None, prefix: {x.secrets}")
        print(f"out: {x.secrets}")
        with secrets_injection_context(x, mounted_secrets_dir, secrets_prefix):
            print(f"in ALL: {x.secrets}")
        print(f"out: {x.secrets}")


class TestLoadModelFromArtifact:
    pass
