#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from datarobot_storage import get_storage
from datarobot_storage.enums import FileStorageBackend

from datarobot_drum.drum.lazy_loading.constants import BackendType
from datarobot_drum.drum.lazy_loading.constants import EnumEncoder
from datarobot_drum.drum.lazy_loading.constants import LazyLoadingEnvVars
from datarobot_drum.drum.lazy_loading.lazy_loading_handler import LazyLoadingHandler
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository
from datarobot_drum.drum.lazy_loading.schema import S3Credentials


class TestDownloadFromMinIO:
    @pytest.fixture
    def skip_minio_test_if_missing_env_vars(self):
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            pytest.skip("AWS_ACCESS_KEY_ID not set in environment")
        if "AWS_SECRET_ACCESS_KEY" not in os.environ:
            pytest.skip("AWS_SECRET_ACCESS_KEY not set in environment")
        if "MINIO_ENDPOINT_URL" not in os.environ:
            pytest.skip("Minio endpoint URL not set in q")

    @pytest.fixture
    def repository(self, skip_minio_test_if_missing_env_vars):
        return LazyLoadingRepository(
            repository_id="66fbd2e8eb9fe6a36622d52d",
            bucket_name="development",
            credential_id="66fbd2e8eb9fe6a36622d52e",
            endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
            verify_certificate=False,
        )

    @pytest.fixture
    def credentials(self):
        return S3Credentials(
            credential_type=BackendType.S3,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    @pytest.fixture
    def lazy_loading_data(self, tmpdir, repository, credentials):
        """Create a lazy loading data by reading files from the minio bucket"""

        config = LazyLoadingHandler.build_s3_config(repository, credentials)
        storage = get_storage(storage_type=FileStorageBackend.S3, config_dict=config)

        files = []
        bucket_key = "custom-model"
        for filename in storage.list(bucket_key):
            files.append(
                {
                    "remote_path": f"{bucket_key}/{filename}",
                    "local_path": str(tmpdir.join(filename)),
                    "repository_id": repository.repository_id,
                }
            )
        return {"files": files, "repositories": [repository.dict()]}

    @pytest.fixture
    def repository_credential_data(self, repository, credentials):
        return {str(repository.credential_id): credentials.dict()}

    @pytest.fixture
    def patch_lazy_loading_env_vars(
        self,
        repository,
        lazy_loading_data,
        repository_credential_data,
        skip_minio_test_if_missing_env_vars,
    ):
        credential_id = repository.credential_id
        credential_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        credential_env_key = f"{credential_prefix}_{credential_id.upper()}"
        with patch.dict(
            os.environ,
            {
                LazyLoadingEnvVars.get_lazy_loading_data_key(): json.dumps(lazy_loading_data),
                credential_env_key: json.dumps(
                    repository_credential_data[credential_id], cls=EnumEncoder
                ),
            },
        ):
            yield

    @pytest.mark.usefixtures("patch_lazy_loading_env_vars", "skip_minio_test_if_missing_env_vars")
    def test_download_from_minio_success(self, lazy_loading_data, repository_credential_data):
        handler = LazyLoadingHandler()
        handler.download_lazy_loading_files()
        assert handler.is_lazy_loading_available
        for file in lazy_loading_data["files"]:
            assert Path(file["local_path"]).exists()
