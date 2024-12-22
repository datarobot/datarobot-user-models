#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import copy
import json
import os
from unittest.mock import patch

import pytest

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter
from datarobot_drum.drum.enum import TargetType
from datarobot_drum.drum.language_predictors.python_predictor.python_predictor import (
    PythonPredictor,
)
from datarobot_drum.drum.lazy_loading.constants import BackendType
from datarobot_drum.drum.lazy_loading.constants import EnumEncoder
from datarobot_drum.drum.lazy_loading.constants import LazyLoadingEnvVars
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingData
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingFile
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository
from datarobot_drum.drum.lazy_loading.schema import S3Credentials


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
class TestPythonPredictorConfigure:
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
        predictor.configure(base_configure_params)

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
        predictor.configure(base_configure_params)

        mock_load_model_from_artifact.assert_called_once_with(
            user_secrets_mount_path=mount_path,
            user_secrets_prefix=prefix,
        )


@pytest.mark.usefixtures("mock_load_model_from_artifact")
@pytest.mark.parametrize("has_chat_hook", [False, True])
def test_supports_chat(base_configure_params, has_chat_hook):
    with patch.object(PythonModelAdapter, "has_custom_hook") as mock_has_custom_hook:
        mock_has_custom_hook.return_value = has_chat_hook

        predictor = PythonPredictor()
        predictor.configure(base_configure_params)

        assert predictor.supports_chat() == has_chat_hook


class TestPythonPredictorLazyLoading:
    @pytest.fixture
    def lazy_loading_env_vars(self):
        env_data = {}
        credential_ids = ["66fbd2e8eb9fe6a36622d52e", "66fbd2e8eb9fe6a36622d52f"]
        lazy_loading_data = LazyLoadingData(
            files=[
                LazyLoadingFile(
                    remote_path="remote/path/large_file1.pkl",
                    local_path="/tmp/large_file1.pkl",
                    repository_id="repo1",
                ),
                LazyLoadingFile(
                    remote_path="remote/path/large_file2.pkl",
                    local_path="/tmp/large_file2.pkl",
                    repository_id="repo2",
                ),
            ],
            repositories=[
                LazyLoadingRepository(
                    repository_id="repo1",
                    bucket_name="bucket1",
                    credential_id=credential_ids[0],
                    endpoint_url="http://minio1:9000",
                    verify_certificate=False,
                ),
                LazyLoadingRepository(
                    repository_id="repo2",
                    bucket_name="bucket2",
                    credential_id=credential_ids[1],
                    endpoint_url="http://minio2:9000",
                    verify_certificate=False,
                ),
            ],
        )
        env_data[LazyLoadingEnvVars.get_lazy_loading_data_key()] = json.dumps(
            lazy_loading_data.dict()
        )
        credential_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        for credential_id in credential_ids:
            credential_env_key = f"{credential_prefix}_{credential_id.upper()}"
            credential_data = S3Credentials(
                credential_type=BackendType.S3,
                aws_access_key_id="dummy_access_key1",
                aws_secret_access_key="dummy_secret_key1",
            )
            env_data[credential_env_key] = json.dumps(credential_data.dict(), cls=EnumEncoder)

        with patch.dict(os.environ, env_data):
            yield

    @pytest.fixture
    def text_generation_model_params(self, essential_language_predictor_init_params):
        with patch.dict(os.environ, {"TARGET_NAME": "Response"}):
            init_params = copy.deepcopy(essential_language_predictor_init_params)
            init_params["target_type"] = TargetType.TEXT_GENERATION.value
            yield init_params

    @pytest.fixture
    def mock_unrelated_methods(self):
        with patch.object(PythonModelAdapter, "load_model_from_artifact"):
            yield

    @pytest.mark.usefixtures("lazy_loading_env_vars", "mock_unrelated_methods")
    def test_lazy_loading_download_is_being_called(self, text_generation_model_params):
        with patch.object(LazyLoadingHandler, "download_lazy_loading_files") as mock_download:
            py_predictor = PythonPredictor()
            py_predictor.configure(text_generation_model_params)
            mock_download.assert_called_once()

    @pytest.mark.usefixtures("mock_unrelated_methods")
    def test_lazy_loading_download_is_not_called(self, text_generation_model_params):
        with patch.object(LazyLoadingHandler, "download_lazy_loading_files") as mock_download:
            py_predictor = PythonPredictor()
            py_predictor.configure(text_generation_model_params)
            mock_download.assert_not_called()
