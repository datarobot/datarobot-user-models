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
from abc import ABC
from abc import abstractmethod


class StorageFileDownloadHelper(ABC):
    def __init__(self, storage_credentials):
        self.storage_credentials = storage_credentials

    @abstractmethod
    def get_file_size(self, file_uri):
        """
        Get the file size in bytes for the given file URI.
        """
        pass

    @abstractmethod
    def download_file(self, result_list, file_info, output_dir, lock, buffer_size, verify_checksum):
        """
        Download the file specified by the file URI.
        """
        pass

    @abstractmethod
    def is_uri_directory(self, file_uri):
        """
        Check if the URI is a directory or a file
        """
        pass

    @abstractmethod
    def list_uris_in_directory(self, dir_uri):
        """
        Return a list of all the URIs in the given directory
        """
        pass
