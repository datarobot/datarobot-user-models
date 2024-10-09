#
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc.
#
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
import logging
from abc import ABC
from abc import abstractmethod

from custom_model_runner.datarobot_drum.lazy_loading.progress_percentage import ProgressPercentage
from custom_model_runner.datarobot_drum.lazy_loading.remote_file import RemoteFile
from custom_model_runner.datarobot_drum.lazy_loading.environment_config_helper import (
    handle_credentials_param,
)
from custom_model_runner.datarobot_drum.lazy_loading.storage_utils import has_mandatory_keys

logger = logging.getLogger(__name__)


class FileRepo(ABC):
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def name(self):
        return self._name

    @abstractmethod
    def is_file_exist(self, file_path):
        pass

    @abstractmethod
    def get_file_size(self, file_path):
        pass

    def download_file(self, remote_file: RemoteFile, progress):
        pass


def build_s3_uri(bucket_name: str, file_path: str) -> str:
    return f"s3://{bucket_name}/{file_path}"


class S3FileRepo(FileRepo):
    mandatory_keys = {"type", "bucket_name", "credential_id", "repository_id", "bucket_name"}

    def __init__(self, name, bucket_name, credentials_id=None):
        super().__init__(name)
        self._bucket_name = bucket_name
        self._credentials_id = credentials_id
        storage_credentials = handle_credentials_param(credentials_id)
        # TODO: Add credentials type check
        # TODO: Implement dr-storage client initialization

    def __str__(self):
        return f"{self._name} [s3]:  bucket: {self._bucket_name}, credentials_env: {self._credentials_id}"

    def is_file_exist(self, file_path) -> bool:
        # TODO: implement Head the object to get its metadata, including content length
        raise NotImplementedError

    def get_file_size(self, file_path):
        """
        Get the file size in bytes.
        :param file_path:
        :return: Size in bytes if file exists, None if not exists. Raise Exception otherwise
        """
        # TODO: implement Head the object to get its metadata, including content length
        raise NotImplementedError

    def download_file(self, remote_file: RemoteFile, progress: ProgressPercentage):
        # TODO: implement file download login using dr-storage
        raise NotImplementedError


class RemoteRepos:
    def __init__(self):
        self._repos = {}

    def from_dict(self, repos_dict):
        """
        Build the RemoteRepos instance from a dictionary.
        :param repos_dict:
        :return:
        """
        for repo_dict in repos_dict:
            if "type" not in repo_dict:
                raise KeyError(f"Missing 'type' key in {repo_dict}")

            if repo_dict["type"] == "s3":
                is_valid, missing_fields = has_mandatory_keys(repo_dict, S3FileRepo.mandatory_keys)
                if is_valid:
                    repository_id = repo_dict["repository_id"]
                    repo_obj = S3FileRepo(
                        repo_dict["repository_id"],
                        repo_dict["bucket_name"],
                        repo_dict["credential_id"],
                    )
                else:
                    raise Exception(f"Repo has missing fields for S3 Repo: {missing_fields}")

            else:
                raise Exception(
                    f"Type {repo_dict['type']} is not supported, only S3 repos are currently supported"
                )
            # From This stage on all repo objects are generic
            self._repos[repository_id] = repo_obj
        return self

    def get_remote_repos(self):
        """
        Get all the names of the remote repos.
        :return: list of repo names
        """
        return self._repos.keys()

    def get_remote_repo(self, repo_id) -> FileRepo:
        """
        Get a RemoteRepo instance by id.
        :param repo_id:
        :return: RemoteRepo instance
        """
        if repo_id not in self._repos:
            raise KeyError(f"Repo {repo_id} not found.")
        return self._repos[repo_id]

    def exists(self, repo_name):
        if repo_name in self._repos:
            return True
        return False
