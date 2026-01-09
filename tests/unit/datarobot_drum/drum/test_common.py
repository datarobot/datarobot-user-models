import os

from unittest.mock import patch

import json

import types
from unittest import mock
import pytest

from datarobot_drum.drum.common import setup_otel

from datarobot_drum import RuntimeParameters

from datarobot_drum.runtime_parameters.runtime_parameters_schema import RuntimeParameterTypes


@pytest.fixture
def otel_sdk_enabled():
    namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name("OTEL_SDK_DISABLED")
    runtime_param_env_value = json.dumps(
        {"type": RuntimeParameterTypes.BOOLEAN.value, "payload": False}
    )
    os.environ[namespaced_runtime_param_name] = runtime_param_env_value
    yield
    del os.environ[namespaced_runtime_param_name]


@pytest.fixture
def otel_sdk_disabled():
    namespaced_runtime_param_name = RuntimeParameters.namespaced_param_name("OTEL_SDK_DISABLED")
    runtime_param_env_value = json.dumps(
        {"type": RuntimeParameterTypes.BOOLEAN.value, "payload": True}
    )
    os.environ[namespaced_runtime_param_name] = runtime_param_env_value
    yield
    del os.environ[namespaced_runtime_param_name]


class TestOtel:
    @staticmethod
    def make_options(max_workers=1):
        return types.SimpleNamespace(max_workers=max_workers)

    @staticmethod
    def make_options_no_max_workers():
        # Return an object without max_workers attribute
        return types.SimpleNamespace()

    def test_setup_otel_disabled(self, monkeypatch, otel_sdk_disabled):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        options = self.make_options()
        result = setup_otel(RuntimeParameters, options)
        assert result == (None, None, None)

    def test_setup_otel_not_configured(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        options = self.make_options()
        result = setup_otel(RuntimeParameters, options)
        assert result == (None, None, None)

    def test_setup_otel_configured_with_max_workers(self, monkeypatch, otel_sdk_enabled):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        options = self.make_options(max_workers=2)
        with mock.patch(
            "datarobot_drum.drum.common._setup_otel_tracing", return_value="tracer"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_logging", return_value="logger"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_metrics", return_value="metrics"
        ):
            result = setup_otel(RuntimeParameters, options)
            assert result == ("tracer", "metrics", "logger")

    def test_setup_otel_configured_without_max_workers(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        options = self.make_options_no_max_workers()
        with mock.patch(
            "datarobot_drum.drum.common._setup_otel_tracing", return_value="tracer"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_logging", return_value="logger"
        ), mock.patch(
            "datarobot_drum.drum.common._setup_otel_metrics", return_value="metrics"
        ):
            result = setup_otel(RuntimeParameters, options)
            assert result == ("tracer", "metrics", "logger")
