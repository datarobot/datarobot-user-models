#
#  Copyright 2024 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
import asyncio
import logging
import os
from typing import Dict
from typing import Optional
from urllib.parse import urlsplit

import pydantic
import yaml
from datarobot_storage import get_async_storage
from datarobot_storage.enums import FileStorageBackend

from datarobot_drum.drum.lazy_loading.constants import BackendType
from datarobot_drum.drum.lazy_loading.constants import LazyLoadingEnvVars
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingCommandLineFileContent
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingData
from datarobot_drum.drum.lazy_loading.schema import LazyLoadingRepository
from datarobot_drum.drum.lazy_loading.schema import S3Credentials


logger = logging.getLogger(__name__)


class LazyLoadingHandler:
    def __init__(self):
        self._lazy_loading_data: Optional[LazyLoadingData] = self._load_lazy_loading_data_from_env()
        if self.is_lazy_loading_available:
            self._credentials: Optional[Dict[str, S3Credentials]] = self._load_credentials_from_env(
                self._lazy_loading_data
            )

    @property
    def is_lazy_loading_available(self):
        return self._lazy_loading_data is not None

    @staticmethod
    def _load_lazy_loading_data_from_env():
        json_string = os.environ.get(LazyLoadingEnvVars.get_lazy_loading_data_key())
        if json_string is None:
            return None
        return LazyLoadingData.from_json_string(json_string)

    @staticmethod
    def _load_credentials_from_env(lazy_loading_data: LazyLoadingData):
        credential_env_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        credentials = {}
        for repository in lazy_loading_data.repositories:
            credential_env_key = f"{credential_env_prefix}_{repository.credential_id.upper()}"
            credential_content = os.environ.get(credential_env_key)
            if credential_content is None:
                raise ValueError(
                    f"Missing credential for repository {repository.repository_id}, "
                    f"credential_id: {repository.credential_id}"
                )
            credentials[repository.credential_id] = S3Credentials.parse_raw(credential_content)
        return credentials

    def download_lazy_loading_files(self):
        if not self.is_lazy_loading_available:
            return
        logger.info("Start downloading lazy loading files")
        asyncio.run(self._download_in_parallel())
        logger.info("Lazy loading files have been downloaded")

    async def _download_in_parallel(self):
        repo_backend_storages = {}
        for repository in self._lazy_loading_data.repositories:
            storage = self._get_backend_storage(repository)
            repo_backend_storages[repository.repository_id] = storage

        tasks = []  # List to hold the coroutine tasks
        for file in self._lazy_loading_data.files:
            storage = repo_backend_storages[file.repository_id]
            logger.info(
                "Add downloading task for remote path", extra={"remote_path": file.remote_path}
            )
            download_file_coroutine = storage.get(file.remote_path, file.local_path)
            tasks.append(download_file_coroutine)
        await asyncio.gather(*tasks)

    def _get_backend_storage(self, repository: LazyLoadingRepository):
        credential = self._credentials[repository.credential_id]
        if credential.credential_type == BackendType.S3:
            storage_config = self.build_s3_config(
                repository, self._credentials[repository.credential_id]
            )
            return get_async_storage(FileStorageBackend.S3, storage_config)
        else:
            raise NotImplementedError(f"Unsupported backend type: {credential.credential_type}")

    @staticmethod
    def build_s3_config(repository: LazyLoadingRepository, credentials: S3Credentials):
        config = {
            "AWS_ACCESS_KEY_ID": credentials.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": credentials.aws_secret_access_key,
            "AWS_SESSION_TOKEN": credentials.aws_session_token,
            "S3_BUCKET": repository.bucket_name,
        }
        if repository.endpoint_url:
            parsed_url = urlsplit(repository.endpoint_url)
            config["S3_IS_SECURE"] = True if parsed_url.scheme == "https" else False
            config["S3_VALIDATE_CERTS"] = repository.verify_certificate
            config["S3_HOST"] = parsed_url.hostname
            if parsed_url.port is not None:
                config["S3_PORT"] = parsed_url.port
        return config

    @staticmethod
    def setup_environment_variables_from_values_file(filepath):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
        except yaml.parser.ParserError as exc:
            raise ValueError(f"Invalid lazy loading YAML file content. {str(exc)}")

        try:
            lazy_loading_cli_file_content = LazyLoadingCommandLineFileContent(**data)
        except pydantic.ValidationError as exc:
            raise ValueError(f"Invalid lazy loading content. {str(exc)}")

        lazy_loading_json = lazy_loading_cli_file_content.model_dump_json(exclude={"credentials"})
        os.environ[LazyLoadingEnvVars.get_lazy_loading_data_key()] = lazy_loading_json

        credentials_prefix = LazyLoadingEnvVars.get_repository_credential_id_key_prefix()
        for credentials_entry in lazy_loading_cli_file_content.credentials:
            credentials_key = f"{credentials_prefix}_{credentials_entry.id.upper()}"
            os.environ[credentials_key] = credentials_entry.model_dump_json(exclude={"id"})
