#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
from unittest.mock import patch, Mock, ANY

import pytest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from tests.unit.datarobot_drum.drum.conftest import create_completion, create_completion_chunks
from werkzeug.exceptions import BadRequest


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
            user_secrets_mount_path=None,
            user_secrets_prefix=None,
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
            user_secrets_mount_path=mount_path,
            user_secrets_prefix=prefix,
        )


class TestChat:
    @pytest.fixture
    def mock_mlops(self):
        with patch(
            "datarobot_drum.drum.language_predictors.python_predictor.python_predictor.MLOps"
        ) as mock:
            mlops_instance = Mock()
            mock.return_value = mlops_instance
            yield mlops_instance

    @pytest.fixture
    def python_predictor(self, chat_python_model_adapter, mock_mlops):
        predictor = PythonPredictor()
        params = {
            "target_type": TargetType.TEXT_GENERATION,
            "__custom_model_path__": "/non-existing-path-to-avoid-loading-unwanted-artifacts",
        }

        with patch.dict(os.environ, {"TARGET_NAME": "target"}):
            predictor.mlpiper_configure(params)

        yield predictor

    @pytest.mark.parametrize("response", ["Wrong response", None])
    @pytest.mark.parametrize("stream", [False, True])
    def test_hook_wrong_response_type(
        self, python_predictor, chat_python_model_adapter, stream, response
    ):
        def chat_hook(completion_request, model):
            return response

        chat_python_model_adapter.chat_hook = chat_hook

        with pytest.raises(
            Exception,
            match=r"Expected response to be ChatCompletion or Iterable\[ChatCompletionChunk\].*",
        ):
            response = python_predictor.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": stream,
                }
            )
            [
                chunk for chunk in response
            ]  # Streaming response needs to be consumed for anything to happen

    def test_mlops_init(self, python_predictor, mock_mlops):
        mock_mlops.set_channel_config.called_once_with("spooler_type=API")

        mock_mlops.init.assert_called_once()

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_with_mlops(self, python_predictor, chat_python_model_adapter, mock_mlops, stream):
        def chat_hook(completion_request, model):
            return (
                create_completion_chunks(["How", " are", " you"])
                if stream
                else create_completion("How are you")
            )

        chat_python_model_adapter.chat_hook = chat_hook

        response = python_predictor.chat(
            {
                "model": "any",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "stream": stream,
            }
        )
        if stream:
            [
                chunk for chunk in response
            ]  # Streaming response needs to be consumed for anything to happen

        mock_mlops.report_deployment_stats.assert_called_once()

        mock_mlops.report_predictions_data.assert_called_once_with(
            ANY,
            ["How are you"],
            association_ids=ANY,
            skip_drift_tracking=True,
            skip_accuracy_tracking=True,
        )
        # Compare features dataframe separately as this doesn't play nice with assert_called
        assert mock_mlops.report_predictions_data.call_args.args[0]["prompt"].values[0] == "Hello!"

    @pytest.mark.parametrize("stream", [False, True])
    def test_failing_hook_with_mlops(
        self, python_predictor, chat_python_model_adapter, mock_mlops, stream
    ):
        def chat_hook(completion_request, model):
            raise BadRequest("Error")

        chat_python_model_adapter.chat_hook = chat_hook

        with pytest.raises(BadRequest):
            response = python_predictor.chat(
                {
                    "model": "any",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "stream": stream,
                }
            )

        mock_mlops.report_deployment_stats.assert_not_called()
        mock_mlops.report_predictions_data.assert_not_called()
