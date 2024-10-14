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
import json
import logging
import os
import time
from typing import List

from datarobot_drum.lazy_loading.constants import MLOPS_LAZY_LOADING_DATA_ENV_VARIABLE

from custom_model_runner.datarobot_drum.lazy_loading.progress_percentage import ProgressPercentage
from custom_model_runner.datarobot_drum.lazy_loading.remote_file import RemoteFile
from custom_model_runner.datarobot_drum.lazy_loading.remote_repos import RemoteRepos

from custom_model_runner.datarobot_drum.lazy_loading.storage_utils import (
    calculate_rate,
    get_disk_space,
)

logger = logging.getLogger(__name__)


class RemoteFiles:
    """
    A helper class to download remote files from remote repository.
    This class is not aware of DataRobot App. Any access to DataRobot should be done
    outside of this class.
    """

    def __init__(
        self,
        local_dir,
        deployment_id=None,
        verify_ssl=True,
        mb_to_update=50,
        seconds_to_update=10,
        chunk_size=1024 * 1024,
        nr_processes=1,
        verify_checksum=True,
    ):
        """ """
        self._local_dir = local_dir
        self._remote_files: List[RemoteFile] = []
        self._remote_repos: RemoteRepos
        self._validated = False
        self._total_size_mb = 0
        self._download_start_time = None
        self._download_end_time = None

    def __str__(self):
        s = "Remote Files:\n"
        s += "Total Size: {:.2f} MB\n".format(self._total_size_mb)
        s += "Files:\n"
        for remote_file in self._remote_files:
            s += str(remote_file) + "\n"
        return s

    @property
    def total_size_mb(self):
        return self._total_size_mb

    @property
    def download_time_seconds(self):
        if self._download_end_time is None or self._download_start_time is None:
            return None
        return self._download_end_time - self._download_start_time

    @property
    def download_rate_mb_seconds(self):
        if self.download_time_seconds is None:
            return None
        return self._total_size_mb / self.download_time_seconds

    # def set_remote_repos(self, remote_repos):
    #     self._remote_repos = remote_repos
    #     return self

    def get_remote_files(self):
        return self._remote_files

    def from_env_config(self):
        if MLOPS_LAZY_LOADING_DATA_ENV_VARIABLE not in os.environ:
            raise Exception("Cant find lazy loading environment variable")

        remote_files_config = json.loads(os.environ[MLOPS_LAZY_LOADING_DATA_ENV_VARIABLE])

        if remote_files_config is None:
            raise Exception("Cant load lazy loading config from environment variable")

        repo_dict = remote_files_config["repositories"]

        self._remote_repos = RemoteRepos().from_dict(repo_dict)

        remote_files = remote_files_config["files"]

        for remote_file in remote_files:
            self._remote_files.append(
                RemoteFile(
                    remote_path=remote_file["remote_path"],
                    local_path=remote_file["local_path"],
                    repository_id=remote_file["repository_id"],
                )
            )

        return self

    def validate(self):
        """
        Validate remote files against the remote repository.
        :return: True if valid, else False. An extra return value contains the issue.
        """
        if self._remote_repos is None:
            raise Exception("Remote Repository is None, must provide remote repos object")
        for remote_file in self._remote_files:
            if not self._remote_repos.exists(remote_file.repository_id):
                raise Exception(
                    f"Remote file: {remote_file.remote_path} repo {remote_file.repository_id} does not exist"
                )

        self._validated = True
        return

    def _update_file_info(self, remote_file):
        logger.debug(f"Updating file info: {remote_file.remote_path}")
        repo = self._remote_repos.get_remote_repo(remote_file.repository_id)
        try:
            file_size = repo.get_file_size(remote_file.remote_path)
            if file_size is None:
                remote_file.error_msg = "File does not exist"
                return False
            remote_file.size_bytes = file_size
            self._total_size_mb += remote_file.size_mb
            return True
        except Exception as e:
            remote_file.error_msg = str(e)
            return False

    def update_files_info(self):
        """
        Get various information about all the remote files. This means a connection will be attempted to all repos and
        for each file info like size will be obtained.

        Returns: update_ok, issue list
        """
        logger.debug("Updating remote files info ...")
        update_status = True
        error_list = []
        for remote_file in self._remote_files:
            file_update_status = self._update_file_info(remote_file)
            if not file_update_status:
                update_status = False
                error_list.append(f"{remote_file.remote_path}: " + remote_file.error_msg)
        return update_status, error_list

    def _check_disk_space(self):
        total_mb, used_mb, free_mb = get_disk_space(self._local_dir)
        print(
            f"Total disk space: {total_mb:.1f} MB, used disk space: {used_mb:.1f} MB, free disk space: {free_mb:.1f} MB"
        )
        if self._total_size_mb > free_mb:
            err_msg = f"Error not enough disk space to download: Required: {self._total_size_mb:.1f} MB > Free: {free_mb:.1f} MB"
            raise Exception(err_msg)

    def _prepare_dir_structure(self):
        for remote_file in self._remote_files:
            dir_name = os.path.dirname(remote_file.local_path)
            print("Directory name: ", dir_name)
            os.makedirs(dir_name, exist_ok=True)

    def _update_local_path(self):
        for remote_file in self._remote_files:
            remote_file.local_path = os.path.join(self._local_dir, remote_file.local_path)

    def _download_file(self, remote_file: RemoteFile, progress: ProgressPercentage):
        repo = self._remote_repos.get_remote_repo(remote_file.repository_id)
        progress.set_file(remote_file)
        download_ok = repo.download_file(remote_file, progress)
        if not download_ok:
            return download_ok

        # Verify size:
        if not os.path.exists(remote_file.local_path):
            remote_file.error_msg = "File does not exist on local path: {}".format(
                remote_file.local_path
            )
            return False
        local_size_bytes = os.path.getsize(remote_file.local_path)
        if local_size_bytes != remote_file.size_bytes:
            remote_file.error_msg = (
                "Size of local file {} is not equal to remote file size: {}".format(
                    local_size_bytes, remote_file.size_bytes
                )
            )
            return False

        return True

    def download(self, progress: ProgressPercentage):
        """
        Download all remote files from remote repository.
        :return:
        """

        print("Downloading remote files ...")
        self.validate()
        update_status, error_list = self.update_files_info()
        if update_status is False:
            return False, error_list

        self._update_local_path()
        self._check_disk_space()
        self._prepare_dir_structure()

        self._download_start_time = time.time()
        overall_download_ok = True
        error_list = []
        for remote_file in self._remote_files:
            download_ok = self._download_file(remote_file, progress)
            if download_ok is False:
                overall_download_ok = False
                error_list.append(f"{remote_file.remote_path}: {remote_file.error_msg}")
            progress.done_downloading_file(remote_file)
            print(remote_file)
        self._download_end_time = time.time()

        return overall_download_ok, error_list
