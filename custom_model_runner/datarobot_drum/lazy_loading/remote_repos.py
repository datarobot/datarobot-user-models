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
import os
import time
from abc import ABC
from abc import abstractmethod

import boto3
from botocore import client
from botocore.exceptions import ClientError

from custom_model_runner.datarobot_drum.lazy_loading.constants import AWS_DEFAULT_REGION
from custom_model_runner.datarobot_drum.lazy_loading.progress_percentage import ProgressPercentage
from custom_model_runner.datarobot_drum.lazy_loading.remote_file import RemoteFile
from custom_model_runner.datarobot_drum.lazy_loading.runtime_params_helper import handle_credentials_param
from custom_model_runner.datarobot_drum.lazy_loading.storage_utils import calculate_rate
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
        self._s3 = boto3.client(
            's3',
            endpoint_url=storage_credentials['endpointUrl'] if 'endpointUrl' in storage_credentials else None,
            aws_access_key_id=storage_credentials['awsAccessKeyId'],
            aws_secret_access_key=storage_credentials['awsSecretAccessKey'],
            aws_session_token=(
                storage_credentials['sessionToken']
                if 'sessionToken' in storage_credentials
                else None
            ),
            region_name=(
                storage_credentials['region'] if 'region' in storage_credentials else AWS_DEFAULT_REGION
            ),
            config=client.Config(signature_version='s3v4'),
        )

    def __str__(self):
        return f"{self._name} [s3]:  bucket: {self._bucket_name}, credentials_env: {self._credentials_id}"

    def is_file_exist(self, file_path) -> bool:
        # Head the object to get its metadata, including content length
        try:
            self._s3.head_object(Bucket=self._bucket_name, Key=file_path)
            return True
        except ClientError as e:
            # Check if the exception is a 404 error
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                # Re-raise the exception if it's not a 404 error
                raise

    def get_file_size(self, file_path):
        """
        Get the file size in bytes.
        :param file_path:
        :return: Size in bytes if file exists, None if not exists. Raise Exception otherwise
        """
        # Head the object to get its metadata, including content length
        try:
            response = self._s3.head_object(Bucket=self._bucket_name, Key=file_path)
            # Extract and return the content length (file size)
            file_size = response["ContentLength"]
            return file_size
        except ClientError as e:
            # Check if the exception is a 404 error
            if e.response["Error"]["Code"] == "404":
                return None
            else:
                # Re-raise the exception if it's not a 404 error
                raise

    def download_file(self, remote_file: RemoteFile, progress: ProgressPercentage):
        # TODO: handle buffer_size
        #  def download_file(self, result_list, file_info, output_dir, lock, buffer_size, verify_checksum):

        # print("Bucket: {}, .Object: {}, Output Dir: {}".
        #       format(file_info["bucket_name"], file_info["object_key"], output_dir))
        #
        # result_info = {}
        try:
            logger.debug("Downloading file: {}".format(remote_file.local_path))
            with open(remote_file.local_path, "wb") as file_handle:
                start_time = time.time()
                self._s3.download_fileobj(
                    self._bucket_name,
                    remote_file.remote_path,
                    file_handle,
                    Callback=progress,
                    Config=boto3.s3.transfer.TransferConfig(
                        max_io_queue=1, io_chunksize=1024 * 1024
                    ),
                )

                # # Calculate elapsed time and bandwidth
                end_time = time.time()
                elapsed_time = end_time - start_time

                remote_file.download_status = True
                local_file_size = os.path.getsize(remote_file.local_path)

                remote_file.download_start_time = start_time
                remote_file.download_time = elapsed_time

                logger.debug(
                    "Downloaded: {}. Bandwidth: {:.1f}".format(
                        remote_file.remote_path,
                        calculate_rate(local_file_size, elapsed_time),
                    )
                )
                return True
        except Exception as e:
            err_msg = "Error downloading {}: {}".format(remote_file.remote_path, e)
            logger.error(err_msg)
            remote_file.error = err_msg
            return False


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
                        repo_dict["repository_id"], repo_dict["bucket_name"], repo_dict["credential_id"]
                    )
                else:
                    raise Exception(
                        f"Repo has missing fields for S3 Repo: {missing_fields}"
                    )

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

    def get_remote_repo(self, repo_name) -> FileRepo:
        """
        Get a RemoteRepo instance by name.
        :param repo_name:
        :return: RemoteRepo instance
        """
        if repo_name not in self._repos:
            raise KeyError(f"Repo {repo_name} not found.")
        return self._repos[repo_name]

    def exists(self, repo_name):
        if repo_name in self._repos:
            return True
        return False
