import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pytest
from unittest.mock import patch
from datarobot_storage.amazon import S3Storage

from datarobot_drum.drum.lazy_loading.constants import BackendType
from datarobot_drum.drum.lazy_loading.constants import EnumEncoder
from datarobot_drum.drum.lazy_loading.constants import LazyLoadingEnvVars
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository
from datarobot_drum.drum.lazy_loading.schema import S3Credentials

logger = logging.getLogger(__name__)


class TestDownloadFromS3:
    @pytest.fixture
    def repository(self):
        return LazyLoadingRepository(
            repository_id="66fbd2e8eb9fe6a36622d52d",
            bucket_name="datarobot-rd",
            credential_id="66fbd2e8eb9fe6a36622d52e",
        )

    @pytest.fixture
    def credentials(self):
        return S3Credentials(
            credential_type=BackendType.S3,
            aws_access_key_id="dummy_access_key",
            aws_secret_access_key="dummy_secret_key",
        )

    @pytest.fixture
    def lazy_loading_data(self, tmpdir, repository, credentials):
        bucket_key = "dev/zohar.mizrahi@datarobot.com/dummy_custom_model"
        files = [
            {
                "remote_path": f"{bucket_key}/README.md",
                "local_path": f"{tmpdir}/README.md",
                "repository_id": "66fbd2e8eb9fe6a36622d52d",
            },
            {
                "remote_path": f"{bucket_key}/custom.py",
                "local_path": f"{tmpdir}/custom.py",
                "repository_id": "66fbd2e8eb9fe6a36622d52d",
            },
            {
                "remote_path": f"{bucket_key}/dummy_binary_training.csv",
                "local_path": f"{tmpdir}/dummy_binary_training.csv",
                "repository_id": "66fbd2e8eb9fe6a36622d52d",
            },
            {
                "remote_path": f"{bucket_key}/memory_consumer.py",
                "local_path": f"{tmpdir}/memory_consumer.py",
                "repository_id": "66fbd2e8eb9fe6a36622d52d",
            },
        ]
        return {"files": files, "repositories": [repository.dict()]}

    @pytest.fixture
    def repository_credential_data(self, repository, credentials):
        return {str(repository.credential_id): credentials.dict()}

    @pytest.fixture
    def patch_lazy_loading_env_vars(
        self, repository, lazy_loading_data, repository_credential_data
    ):
        credential_id = repository.credential_id
        credential_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        credential_env_key = f"{credential_prefix}_{credential_id.upper()}"
        env_data = {
            LazyLoadingEnvVars.get_lazy_loading_data_key(): json.dumps(lazy_loading_data),
            credential_env_key: json.dumps(
                repository_credential_data[credential_id], cls=EnumEncoder
            ),
        }
        with patch.dict(os.environ, env_data):
            yield

    @pytest.fixture
    def patch_empty_lazy_loading_env_vars(self):
        with patch.dict(os.environ, {}):
            yield

    @pytest.fixture
    def set_logging_level_to_info(self):
        original_level = logger.level
        logger.setLevel(logging.INFO)
        yield
        logger.setLevel(original_level)

    @pytest.mark.usefixtures("patch_lazy_loading_env_vars", "set_logging_level_to_info")
    def test_download_success(self, lazy_loading_data, repository_credential_data, tmpdir, caplog):
        """
        The test validates that the download process is successful and the files are
        downloaded. It mocks the S3Storage.get method to simulate the download process, which is
        a synchronous operation. This operation is being executed using async.io, which is covered
        by the datarobot_storage library.
        Note that the use of tim.sleep(0.01) is required to test the logging order.
        """

        def mock_s3_storage_get_method(remote_path, local_path):
            logger.info(
                "Start downloading file from  S3.",
                extra={"remote_path": remote_path, "local_path": local_path},
            )
            # Simulate a download delay - it's required to test the logging order
            time.sleep(0.01)
            with open(local_path, "w") as file:
                file.write("dummy content")
            logger.info(
                "Finished downloading file from S3.",
                extra={"remote_path": remote_path, "local_path": local_path},
            )

        with patch.object(S3Storage, "get", side_effect=mock_s3_storage_get_method):
            handler = LazyLoadingHandler()
            handler.download_lazy_loading_files()
        assert handler.is_lazy_loading_available
        for file in lazy_loading_data["files"]:
            assert Path(file["local_path"]).exists()
        # Validate that all the 'Start downloading file from  S3.' come before
        # the 'Finished downloading file from S3.'
        start_messages_counter = 0
        finished_messages_counter = 0
        for record in caplog.records:
            if record.message.startswith("Start downloading file from  S3."):
                start_messages_counter += 1
                assert finished_messages_counter == 0
            elif record.message.startswith("Finished downloading file from S3."):
                finished_messages_counter += 1
        assert start_messages_counter == len(lazy_loading_data["files"])
        assert start_messages_counter == finished_messages_counter

    @pytest.mark.usefixtures("patch_empty_lazy_loading_env_vars")
    def test_download_not_run_when_lazy_loading_data_not_available_in_env(self):
        with patch.object(LazyLoadingHandler, "_download_in_parallel") as download_in_parallel_mock:
            handler = LazyLoadingHandler()
            handler.download_lazy_loading_files()
            assert not handler.is_lazy_loading_available
            download_in_parallel_mock.assert_not_called()
