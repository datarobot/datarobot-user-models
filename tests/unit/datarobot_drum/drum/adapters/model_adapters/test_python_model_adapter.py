#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
from unittest.mock import Mock

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


class TestingPythonModelAdapter(PythonModelAdapter):
    def __init__(self):
        super().__init__(Mock(), Mock())
        self._custom_task_class = FakeCustomTask

    @property
    def custom_task_instance(self) -> FakeCustomTask:
        return self._custom_task_class_instance


class TestFit:
    def test_thing(self):
        adapter = TestingPythonModelAdapter()
        adapter.fit(Mock(), Mock(), Mock())

        instance = adapter.custom_task_instance
        print("\nONE")
        print(instance.fit_calls)
        print(instance.fit_secrets)
        print(instance.save_calls)
        print(instance.save_secrets)


class TestLoadModelFromArtifact:
    pass
