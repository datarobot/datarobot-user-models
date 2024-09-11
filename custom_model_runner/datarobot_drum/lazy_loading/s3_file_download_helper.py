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
import urllib

from progress_percentage import ProgressPercentage
from storage_file_download_helper import StorageFileDownloadHelper
#from utils import calculate_rate
from storage_utils import urlparse
#from datarobot_custom_code.utils import verify_file_integrity

logger = logging.getLogger(__name__)


class S3FileDownloadHelper(StorageFileDownloadHelper):
    def __init__(self):

        self._s3 = None
        # TODO: implement dr-storage client

    def get_file_size(self, file_uri):
        # TODO: implement dr-storage get_file_size
        file_size = 1000
        return file_size

    def download_file(self, result_list, file_info, output_dir, lock, buffer_size, verify_checksum):
        logger.debug(
            "Bucket: {}, .Object: {}, Output Dir: {}".format(
                file_info["bucket_name"], file_info["object_key"], output_dir
            )
        )
        result_info = {}
        try:
            logger.debug("Downloading file: {}".format(file_info["local_file"]))
            # Initialize variables for bandwidth calculation
            start_time = time.time()

            # TODO: Implement Downloading the file
            end_time = time.time()
            elapsed_time = end_time - start_time

            #TODO: implement file file integrity check
            #result_info["download_ok"] = verify_file_integrity(file_info, verify_checksum)
            local_file_size = os.path.getsize(file_info["local_file"])
            logger.debug(f"Elapsed time: {elapsed_time}")
            logger.debug(f"File size: {local_file_size}")

            result_info["index"] = file_info["index"]
            result_info["elapsed_time"] = elapsed_time
            result_info["total_time_sec"] = elapsed_time
            #TODO: Implement Downloading rate calc
            download_rate = 100.0
            #download_rate = calculate_rate(local_file_size, elapsed_time)
            #result_info["rate_mb_sec"] = download_rate
            logger.debug(
                "Downloaded: {}. Bandwidth: {:.1f}".format(
                    file_info["object_key"],
                    download_rate,
                )
            )
            result_list.append(result_info)
            return 0
        except Exception as e:
            logger.error("Error downloading {}: {}".format(file_info["object_key"], e))
            return 0

    def is_uri_directory(self, file_uri):
        parsed_uri = urllib.parse.urlparse(file_uri)
        bucket_name = parsed_uri.netloc
        object_key = parsed_uri.path.lstrip("/")
        # Use head_object to check if the S3 URI points to an object or a directory
        try:
            self._s3.head_object(Bucket=bucket_name, Key=object_key)
            return False  # File exists, it's not a directory
        except self._s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.error("Directory does not exist")
                return True  # Directory does not exist, it's a directory
            else:
                raise  # Other errors should be raised

    def list_uris_in_directory(self, dir_uri):
        # TODO: implement Parse the S3 directory URI to extract bucket name and prefix (directory path)
        file_uris = []
        return file_uris
