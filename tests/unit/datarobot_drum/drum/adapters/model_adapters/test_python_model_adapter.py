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
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any
from unittest.mock import Mock, patch
import textwrap

import pytest

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)

from openai.types.chat.chat_completion import Choice

from datarobot_drum.custom_task_interfaces.user_secrets import (
    secrets_factory,
    SecretsScrubberFilter,
    TextStreamSecretsScrubber,
    AbstractSecret,
    reset_outputs_to_allow_secrets,
)
from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import (
    TargetType,
    MODERATIONS_HOOK,
    MODERATIONS_HOOK_MODULE,
    MODERATIONS_LIBRARY_PACKAGE,
    CustomHooks,
)
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

    def test_load_chat_hook_without_score(self):
        adapter = PythonModelAdapter(Mock(), Mock())
        adapter._custom_hooks[CustomHooks.LOAD_MODEL] = lambda _: True
        adapter._custom_hooks[CustomHooks.CHAT] = lambda model, completion_request: None

        assert adapter.load_model_from_artifact()

    def test_load_chat_hook_with_score(self):
        adapter = PythonModelAdapter(Mock(), Mock())
        adapter._custom_hooks[CustomHooks.LOAD_MODEL] = lambda _: True
        adapter._custom_hooks[CustomHooks.CHAT] = lambda _: None
        adapter._custom_hooks[CustomHooks.SCORE] = lambda _: None

        assert adapter.load_model_from_artifact()


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
    def extra_model_output_df(self, num_rows):
        return pd.DataFrame({"col1": list(range(num_rows)), "col2": list(range(num_rows))})

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

    @pytest.fixture
    def text_generation_target_name(self):
        return "Query Response"

    @pytest.fixture
    def text_generation_df(self, num_rows, text_generation_target_name):
        with patch.dict(os.environ, {"TARGET_NAME": text_generation_target_name}):
            yield pd.DataFrame(
                {
                    text_generation_target_name: [
                        f"Some LLM response - {random.random()}" for _ in range(num_rows)
                    ]
                }
            )

    def test_split_result_of_binary_classification(self, binary_df):
        (
            pred_df,
            extra_model_output,
        ) = PythonModelAdapter(
            Mock(), TargetType.BINARY
        )._split_to_predictions_and_extra_model_output(
            binary_df, request_labels=binary_df.columns.tolist()
        )
        assert pred_df.equals(binary_df)
        assert extra_model_output is None

    def test_split_result_of_binary_classification_with_extra(
        self, binary_df, extra_model_output_df
    ):
        combined_df = binary_df.join(extra_model_output_df)
        (
            pred_df,
            extra_model_output_response,
        ) = PythonModelAdapter(
            Mock(), TargetType.BINARY
        )._split_to_predictions_and_extra_model_output(
            combined_df, request_labels=binary_df.columns.tolist()
        )
        assert pred_df.equals(binary_df)
        assert extra_model_output_response.equals(extra_model_output_df)

    def test_split_result_of_multiclass(self, multiclass_df):
        (
            pred_df,
            extra_model_output,
        ) = PythonModelAdapter(
            Mock(), TargetType.MULTICLASS
        )._split_to_predictions_and_extra_model_output(
            multiclass_df, request_labels=multiclass_df.columns.tolist()
        )
        assert pred_df.equals(multiclass_df)
        assert extra_model_output is None

    def test_split_result_of_multiclass_with_extra(self, multiclass_df, extra_model_output_df):
        combined_df = multiclass_df.join(extra_model_output_df)
        (
            pred_df,
            extra_model_output_response,
        ) = PythonModelAdapter(
            Mock(), TargetType.MULTICLASS
        )._split_to_predictions_and_extra_model_output(
            combined_df, request_labels=multiclass_df.columns.tolist()
        )
        assert pred_df.equals(multiclass_df)
        assert extra_model_output_response.equals(extra_model_output_df)

    def test_split_result_of_regression(self, regression_df):
        (
            pred_df,
            extra_model_output,
        ) = PythonModelAdapter(
            Mock(), TargetType.REGRESSION
        )._split_to_predictions_and_extra_model_output(regression_df, request_labels=None)
        assert pred_df.equals(regression_df)
        assert extra_model_output is None

    def test_split_result_of_regression_with_extra(self, regression_df, extra_model_output_df):
        combined_df = regression_df.join(extra_model_output_df)
        (
            pred_df,
            extra_model_output_response,
        ) = PythonModelAdapter(
            Mock(), TargetType.REGRESSION
        )._split_to_predictions_and_extra_model_output(combined_df, request_labels=None)
        assert pred_df.equals(regression_df)
        assert extra_model_output_response.equals(extra_model_output_df)

    def test_split_result_of_text_generation(self, text_generation_df):
        (
            pred_df,
            extra_model_output,
        ) = PythonModelAdapter(
            Mock(), TargetType.TEXT_GENERATION
        )._split_to_predictions_and_extra_model_output(text_generation_df, request_labels=None)
        assert pred_df.equals(text_generation_df)
        assert extra_model_output is None

    def test_split_result_of_text_generation_with_exta_model_output(
        self, text_generation_df, extra_model_output_df
    ):
        combined_df = text_generation_df.join(extra_model_output_df)
        (
            pred_df,
            extra_model_output_response,
        ) = PythonModelAdapter(
            Mock(), TargetType.TEXT_GENERATION
        )._split_to_predictions_and_extra_model_output(combined_df, request_labels=None)
        assert pred_df.equals(text_generation_df)
        assert extra_model_output_response.equals(extra_model_output_df)

    def test_text_generation_with_exta_model_output_and_redundant_quotation_marks_in_target(
        self, text_generation_df, extra_model_output_df, text_generation_target_name
    ):
        target_name_with_quotation_marks = f'"{text_generation_target_name}"'
        self._test_and_verify_extra_model_output_with_redundant_quotation_marks(
            target_name_with_quotation_marks, text_generation_df, extra_model_output_df
        )

    @staticmethod
    def _test_and_verify_extra_model_output_with_redundant_quotation_marks(
        target_name_with_quotation_marks, text_generation_df, extra_model_output_df
    ):
        with patch.dict(os.environ, {"TARGET_NAME": target_name_with_quotation_marks}):
            combined_df = text_generation_df.join(extra_model_output_df)
            (
                pred_df,
                extra_model_output_response,
            ) = PythonModelAdapter(
                Mock(), TargetType.TEXT_GENERATION
            )._split_to_predictions_and_extra_model_output(combined_df, request_labels=None)
            assert pred_df.equals(text_generation_df)
            assert extra_model_output_response.equals(extra_model_output_df)

    def test_text_generation_with_exta_model_output_and_redundant_quotation_marks_in_both_target_and_df(
        self, text_generation_df, extra_model_output_df, text_generation_target_name
    ):
        target_name_with_quotation_marks = f'"{text_generation_target_name}"'
        text_generation_df.rename(
            columns={text_generation_target_name: target_name_with_quotation_marks}
        )
        self._test_and_verify_extra_model_output_with_redundant_quotation_marks(
            target_name_with_quotation_marks, text_generation_df, extra_model_output_df
        )


class TestPythonModelAdapterInitialization:
    """Use cases to test the Python adapter initialization"""

    def test_valid_initialization_for_text_generation(self):
        text_generation_target_name = "Response"
        with patch.dict(os.environ, {"TARGET_NAME": text_generation_target_name}):
            adapter = PythonModelAdapter(Mock(), TargetType.TEXT_GENERATION)
            assert adapter._target_name == text_generation_target_name

    def test_invalid_initialization_for_text_generation(self):
        os.environ.pop("TARGET_NAME", None)
        with pytest.raises(ValueError, match="Unexpected empty target name"):
            PythonModelAdapter(Mock(), TargetType.TEXT_GENERATION)

    def test_valid_initialization_for_vector_database(self):
        vdb_target_name = "relevant"
        with patch.dict(os.environ, {"TARGET_NAME": vdb_target_name}):
            adapter = PythonModelAdapter(Mock(), TargetType.VECTOR_DATABASE)
            assert adapter._target_name == vdb_target_name

    def test_invalid_initialization_for_vector_database(self):
        os.environ.pop("TARGET_NAME", None)
        with pytest.raises(ValueError, match="Unexpected empty target name"):
            PythonModelAdapter(Mock(), TargetType.VECTOR_DATABASE)

    def test_valid_initialization_for_agentic_workflow(self):
        target_name = "Response"
        with patch.dict(os.environ, {"TARGET_NAME": target_name}):
            adapter = PythonModelAdapter(Mock(), TargetType.AGENTIC_WORKFLOW)
            assert adapter._target_name == target_name

    def test_invalid_initialization_for_agentic_workflow(self):
        os.environ.pop("TARGET_NAME", None)
        with pytest.raises(ValueError, match="Unexpected empty target name"):
            PythonModelAdapter(Mock(), TargetType.AGENTIC_WORKFLOW)


def set_moderations_lib_content(path: Path, content: str):
    # Create dummy datarobot dome package with the drum_integration.py
    mod_dir = path / MODERATIONS_LIBRARY_PACKAGE
    mod_dir.mkdir(parents=True, exist_ok=True)
    mod_init_py = mod_dir / "__init__.py"
    mod_init_py.touch(exist_ok=True)
    mod_hook_file = mod_dir / f"{MODERATIONS_HOOK}.py"
    mod_hook_file.write_text(content)


def remove_moderations_lib_content(path: Path):
    """Removes the moderations subdirectory from the specified path."""
    mod_dir = path / MODERATIONS_LIBRARY_PACKAGE
    if not mod_dir.exists() or not mod_dir.is_dir():
        return

    shutil.rmtree(mod_dir)


def alternate_case(s: str) -> str:
    """Utility to alternate the case of the provided string."""
    return "".join([char.upper() if idx % 2 == 0 else char.lower() for idx, char in enumerate(s)])


def build_chat_completion(message):
    """Builds a chat completion from the provide message"""
    return ChatCompletion(
        id="id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message),
            )
        ],
        created=123,
        model="model",
        object="chat.completion",
    )


def score_alt_case(data, model, **kwargs):
    """Dummy score method -- alternate case to verify it is called"""
    data["completion"] = pd.array([alternate_case(s) for s in data["text"].array])
    return data


def score_pass(data, model, **kwargs):
    """Dummy score method -- pass through unchanged"""
    return data


def chat_alt_case(completion_create_params, model, association_id=None):
    """Dummy chat method -- alternate case to verify it is called"""
    text = alternate_case(completion_create_params["messages"][-1]["content"])
    return build_chat_completion(text)


def chat_pass(completion_create_params, model, association_id=None):
    """Dummy chat method -- pass through text unchanged"""
    text = completion_create_params["messages"][-1]["content"]
    return build_chat_completion(text)


@contextlib.contextmanager
def mock_moderation_content(path: Path, content: str):
    """
    Sets the moderations content in the provided path, makes adjustments to find it, and
    cleans up the files and modules following test execution.
    """

    # remove any currently loaded moderations libraries
    sys.modules.pop(MODERATIONS_HOOK_MODULE, None)
    sys.modules.pop(MODERATIONS_LIBRARY_PACKAGE, None)

    # put provided path in the search path for modules
    sys.path.insert(0, str(path))

    # set the content to the provided value
    set_moderations_lib_content(path, content)
    try:
        yield  # let the test run here
    finally:
        # remove the moderations subdirectory and remove it from search path
        remove_moderations_lib_content(path)
        sys.path.remove(str(path))

        # unload any moderation modules, so they don't get used by another test
        sys.modules.pop(MODERATIONS_HOOK_MODULE, None)
        sys.modules.pop(MODERATIONS_LIBRARY_PACKAGE, None)


class TestModerationPipeline:
    """Mocks the moderations pipeline to test interactions with DRUM."""

    def __init__(self, *args, **kwargs):
        self.score_call_count = 0
        self.chat_call_count = 0

    def score(self, input_df: pd.DataFrame, model, drum_score_fn, **kwargs):
        """Score function turns the completions to uppercase and runs the DRUM score function."""
        self.score_call_count += 1
        input_df["completion"] = input_df["text"].str.upper()
        return drum_score_fn(input_df, model, **kwargs)

    def chat(
        self,
        completion_create_params,
        model,
        drum_chat_fn,
        association_id: str = None,
        **kwargs,
    ):
        """Chat function turns the message to uppercase and runs the DRUM chat function."""
        self.chat_call_count += 1
        prompt = completion_create_params["messages"][-1]["content"].upper()
        completion_create_params["messages"][-1]["content"] = prompt
        return drum_chat_fn(completion_create_params, model)


class TestPythonModelAdapterWithGuards:
    """Use cases to test the moderation integration with DRUM"""

    @pytest.mark.parametrize(
        ["target_type"],
        [
            pytest.param(TargetType.TEXT_GENERATION, id="textgen"),
            pytest.param(TargetType.AGENTIC_WORKFLOW, id="agentic"),
            pytest.param(TargetType.VECTOR_DATABASE, id="vectordb"),
        ],
    )
    def test_loading_moderations_pipeline(self, target_type, tmp_path):
        moderation_content = """
        from unittest.mock import Mock

        def moderation_pipeline_factory(target_type):
            return Mock()
        """
        target_name = "completion"
        with (
            patch.dict(os.environ, {"TARGET_NAME": target_name}),
            mock_moderation_content(tmp_path, textwrap.dedent(moderation_content)),
        ):
            adapter = PythonModelAdapter(tmp_path, target_type)
            assert adapter._mod_pipeline is not None

    def test_vdb_moderation_pipeline(self, tmp_path):
        df = pd.DataFrame({"text": ["abc", "def"]})
        data = bytes(df.to_csv(index=False), encoding="utf-8")
        target_name = "completion"
        with patch.dict(os.environ, {"TARGET_NAME": target_name}):
            adapter = PythonModelAdapter(tmp_path, TargetType.VECTOR_DATABASE)
            adapter._mod_pipeline = TestModerationPipeline()
            adapter._custom_hooks["score"] = score_pass

            result = adapter.predict(binary_data=data)
            assert result is not None
            assert adapter._mod_pipeline.score_call_count == 1

    @pytest.mark.parametrize(
        ["pipeline_present", "score_fn", "pipeline_count", "expected_predictions"],
        [
            pytest.param(True, score_pass, 1, np.array([["ABC"], ["DEF"]]), id="all-pass"),
            pytest.param(False, score_pass, 0, np.array([["abc"], ["def"]]), id="no-pipe-pass"),
            pytest.param(True, score_alt_case, 1, np.array([["AbC"], ["DeF"]]), id="all-alt"),
            pytest.param(False, score_alt_case, 0, np.array([["AbC"], ["DeF"]]), id="no-pipe-alt"),
        ],
    )
    def test_llm_moderation_pipeline_score(
        self, tmp_path, pipeline_present, score_fn, pipeline_count, expected_predictions
    ):
        df = pd.DataFrame({"text": ["abc", "def"]})
        data = bytes(df.to_csv(index=False), encoding="utf-8")
        target_name = "completion"
        with patch.dict(os.environ, {"TARGET_NAME": target_name}):
            adapter = PythonModelAdapter(tmp_path, TargetType.TEXT_GENERATION)
            if pipeline_present:
                adapter._mod_pipeline = TestModerationPipeline()
            # NOTE: without score hook present, uses a legacy path (that is not setup)
            adapter._custom_hooks["score"] = score_fn

            response = adapter.predict(binary_data=data)
            assert np.all(response.predictions == expected_predictions)
            assert pipeline_count == (
                adapter._mod_pipeline.score_call_count if adapter._mod_pipeline else 0
            )

    @pytest.mark.parametrize(
        ["pipeline_present", "chat_fn", "pipeline_count", "expected_completion"],
        [
            pytest.param(True, chat_pass, 1, "HELLO THERE", id="all-pass"),
            pytest.param(False, chat_pass, 0, "Hello there", id="no-pipe-pass"),
            pytest.param(True, chat_alt_case, 1, "HeLlO ThErE", id="all-alt"),
            pytest.param(False, chat_alt_case, 0, "HeLlO ThErE", id="no-pipe-alt"),
        ],
    )
    def test_llm_moderation_pipeline_chat(
        self, tmp_path, pipeline_present, chat_fn, pipeline_count, expected_completion
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there"},
        ]
        target_name = "completion"
        with patch.dict(os.environ, {"TARGET_NAME": target_name}):
            adapter = PythonModelAdapter(tmp_path, TargetType.TEXT_GENERATION)
            if pipeline_present:
                adapter._mod_pipeline = TestModerationPipeline()
            adapter._custom_hooks["chat"] = chat_fn

            response = adapter.chat({"messages": messages}, None, "association_id")
            # Even if guard pipeline exists - moderation chat wrapper does not exist, so invoke
            # only user chat method
            assert response.choices[0].message.content == expected_completion
            assert pipeline_count == (
                adapter._mod_pipeline.chat_call_count if adapter._mod_pipeline else 0
            )
