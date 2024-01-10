#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import contextlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any
from unittest.mock import Mock, patch

import pytest

from datarobot_drum.custom_task_interfaces.user_secrets import (
    secrets_factory,
    SecretsScrubberFilter,
    TextStreamSecretsScrubber,
    AbstractSecret,
    reset_outputs_to_allow_secrets,
)
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.exceptions import DrumCommonException


def get_all_logging_filters():
    root_logger = logging.root.manager.root
    all_loggers = [logger for logger in logging.root.manager.loggerDict.values()]
    all_loggers.append(root_logger)
    return {logger.name: logger.filters[:] for logger in all_loggers if hasattr(logger, "filters")}


def assert_no_secrets_filters(filters_dict: Dict[str, list]):
    for logger_name, logger_filters in filters_dict.items():
        assert all(not isinstance(el, SecretsScrubberFilter) for el in logger_filters), logger_name


def assert_secrets_filters(filters_dict: Dict[str, list], secrets: Dict[str, AbstractSecret]):
    expected_filter = SecretsScrubberFilter(secrets.values())
    for logger_name, logger_filters in filters_dict.items():
        assert expected_filter in logger_filters, logger_name


@dataclass
class Outputs:
    out: Any
    err: Any

    @classmethod
    def from_env(cls):
        return cls(out=sys.stdout, err=sys.stderr)


def assert_unwrapped_outputs(outputs: Outputs):
    assert not isinstance(outputs.out, TextStreamSecretsScrubber)
    assert not isinstance(outputs.err, TextStreamSecretsScrubber)


def assert_wrapped_outputs(outputs: Outputs, secrets: Dict[str, AbstractSecret]):
    assert isinstance(outputs.out, TextStreamSecretsScrubber)
    assert isinstance(outputs.err, TextStreamSecretsScrubber)
    expected_out = TextStreamSecretsScrubber(secrets=secrets.values(), stream=outputs.out.stream)
    expected_err = TextStreamSecretsScrubber(secrets=secrets.values(), stream=outputs.err.stream)
    assert outputs.out == expected_out
    assert outputs.err == expected_err


class FakeCustomTask:
    def __init__(self):
        self.secrets = None
        self.fit_secrets = Mock()
        self.save_secrets = Mock()
        self.fit_calls = []
        self.save_calls = []
        self.fit_outputs = Outputs(None, None)
        self.save_outputs = Outputs(None, None)
        self.fit_logging_filters = Mock()
        self.save_logging_filters = Mock()
        self.load_args = []

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))
        self.fit_secrets = self.secrets
        self.fit_outputs = Outputs.from_env()
        self.fit_logging_filters = get_all_logging_filters()

    def save(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))
        self.save_secrets = self.secrets
        self.save_outputs = Outputs.from_env()
        self.save_logging_filters = get_all_logging_filters()

    @classmethod
    def load(cls, *args, **kwargs):
        instance = cls()
        instance.load_args = (args, kwargs)
        return instance


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
        "FROM_ENV": {"credential_type": "basic", "username": "from-env", "password": "env-password"}
    }


@pytest.fixture
def mounted_secret():
    return {
        "FROM_MOUNTED": {
            "credential_type": "basic",
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
        assert_no_secrets_filters(instance.fit_logging_filters)
        assert_no_secrets_filters(instance.save_logging_filters)
        assert_unwrapped_outputs(instance.fit_outputs)
        assert_unwrapped_outputs(instance.save_outputs)

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

        expected_secrets = {k: secrets_factory(v) for k, v in mounted_secret.items()}
        assert instance.fit_secrets == expected_secrets
        assert instance.save_secrets is None
        assert_wrapped_outputs(instance.fit_outputs, expected_secrets)
        assert_unwrapped_outputs(instance.save_outputs)
        assert_secrets_filters(instance.fit_logging_filters, expected_secrets)
        assert_no_secrets_filters(instance.save_logging_filters)

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
        expected_secrets = {k: secrets_factory(v) for k, v in env_secret.items()}
        assert instance.fit_secrets == expected_secrets
        assert instance.save_secrets is None
        assert_wrapped_outputs(instance.fit_outputs, expected_secrets)
        assert_unwrapped_outputs(instance.save_outputs)
        assert_secrets_filters(instance.fit_logging_filters, expected_secrets)
        assert_no_secrets_filters(instance.save_logging_filters)


@pytest.mark.usefixtures("secrets", "reset_outputs")
class TestLoadModelFromArtifact:
    @pytest.fixture
    def reset_outputs(self):
        reset_outputs_to_allow_secrets()
        yield
        reset_outputs_to_allow_secrets()

    def test_load_with_no_secrets(self):
        assert_unwrapped_outputs(Outputs.from_env())
        assert_no_secrets_filters(get_all_logging_filters())

        model_dir = Mock()
        adapter = TestingPythonModelAdapter(model_dir, Mock())
        assert adapter.custom_task_instance is None
        adapter.load_model_from_artifact(
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
        )
        instance = adapter.custom_task_instance
        assert instance.secrets == {}
        assert_unwrapped_outputs(Outputs.from_env())
        assert_no_secrets_filters(get_all_logging_filters())

        assert instance.load_args == ((model_dir,), {})

    def test_load_with_mount_secrets(self, mounted_secret, mounted_secrets_dir):
        assert_unwrapped_outputs(Outputs.from_env())
        assert_no_secrets_filters(get_all_logging_filters())

        adapter = TestingPythonModelAdapter(Mock(), Mock())
        assert adapter.custom_task_instance is None
        adapter.load_model_from_artifact(
            user_secrets_mount_path=mounted_secrets_dir,
            user_secrets_prefix=None,
        )
        instance = adapter.custom_task_instance
        expected_secrets = {k: secrets_factory(v) for k, v in mounted_secret.items()}
        assert instance.secrets == expected_secrets
        assert_wrapped_outputs(Outputs.from_env(), expected_secrets)
        assert_secrets_filters(get_all_logging_filters(), expected_secrets)

    def test_load_with_env_secrets(self, env_secret, secrets_prefix):
        assert_unwrapped_outputs(Outputs.from_env())
        assert_no_secrets_filters(get_all_logging_filters())

        adapter = TestingPythonModelAdapter(Mock(), Mock())
        assert adapter.custom_task_instance is None
        adapter.load_model_from_artifact(
            user_secrets_mount_path=None,
            user_secrets_prefix=secrets_prefix,
        )
        instance = adapter.custom_task_instance
        expected_secrets = {k: secrets_factory(v) for k, v in env_secret.items()}
        assert instance.secrets == expected_secrets
        assert_wrapped_outputs(Outputs.from_env(), expected_secrets)
        assert_secrets_filters(get_all_logging_filters(), expected_secrets)


class TestPythonModelAdapterPrivateHelpers:
    def test_multiple_artifacts_detection_negative(self):
        with TemporaryDirectory() as dir_name:
            adapter = TestingPythonModelAdapter(dir_name, Mock())
            # create two files with the same extension
            Path(f"{dir_name}/file1.pkl").touch()
            Path(f"{dir_name}/file2.pkl").touch()
            with pytest.raises(DrumCommonException, match="Multiple serialized model files found."):
                adapter._detect_model_artifact_file()


class TestPredictResultSplitter:
    """
    Test the method that takes the predict output DataFrame and splits it to predictions DataFrame
    and extra DataFrame.
    """

    @pytest.fixture
    def num_rows(self):
        return 3

    @pytest.fixture
    def binary_df(self, num_rows):
        positive_probabilities = [float(random.random()) for _ in range(num_rows)]
        negative_probabilities = [1 - v for v in positive_probabilities]
        return pd.DataFrame({"0": negative_probabilities, "1": positive_probabilities})

    @pytest.fixture
    def multiclass_df(self, num_rows):
        class1_values = [0.1 * index for index in range(1, num_rows + 1)]
        class2_values = [0.2 * index for index in range(1, num_rows + 1)]
        class3_values = [1 - (c1 + c2) for c1, c2 in zip(class1_values, class2_values)]
        data = {"class1": class1_values, "class2": class2_values, "class3": class3_values}
        return pd.DataFrame(data)

    @pytest.fixture
    def regression_df(self, num_rows):
        return pd.DataFrame({"Predictions": [random.random() for _ in range(num_rows)]})

    def test_split_result_of_binary_classification(self, binary_df):
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            binary_df, request_labels=binary_df.columns.tolist()
        )
        assert pred_df.equals(binary_df)
        assert passthrough_df is None

    def test_split_result_of_binary_classification_with_extra(self, num_rows, binary_df):
        in_passthrough_df = pd.DataFrame(
            {"col1": list(range(num_rows)), "col2": list(range(num_rows))}
        )
        combined_df = binary_df.join(in_passthrough_df)
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            combined_df, request_labels=binary_df.columns.tolist()
        )
        assert pred_df.equals(binary_df)
        assert passthrough_df.equals(in_passthrough_df)

    def test_split_result_of_multiclass(self, multiclass_df):
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            multiclass_df, request_labels=multiclass_df.columns.tolist()
        )
        assert pred_df.equals(multiclass_df)
        assert passthrough_df is None

    def test_split_result_of_multiclass_with_extra(self, num_rows, multiclass_df):
        in_passthrough_df = pd.DataFrame(
            {"ext1": list(range(num_rows)), "ext2": list(range(num_rows))}
        )
        combined_df = multiclass_df.join(in_passthrough_df)
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            combined_df, request_labels=multiclass_df.columns.tolist()
        )
        assert pred_df.equals(multiclass_df)
        assert passthrough_df.equals(in_passthrough_df)

    def test_split_result_of_regression(self, regression_df):
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            regression_df, request_labels=None
        )
        assert pred_df.equals(regression_df)
        assert passthrough_df is None

    def test_split_result_of_regression_with_extra(self, num_rows, regression_df):
        in_passthrough_df = pd.DataFrame(
            {"ext1": list(range(num_rows)), "ext2": list(range(num_rows))}
        )
        combined_df = regression_df.join(in_passthrough_df)
        pred_df, passthrough_df = PythonModelAdapter._split_to_predictions_and_passthrough_df(
            combined_df, request_labels=None
        )
        assert pred_df.equals(regression_df)
        assert passthrough_df.equals(in_passthrough_df)
