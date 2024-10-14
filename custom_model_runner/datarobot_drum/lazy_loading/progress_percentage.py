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
import time
from multiprocessing import Lock

from custom_model_runner.datarobot_drum.lazy_loading.remote_file import RemoteFile

logger = logging.getLogger(__name__)


class ProgressPercentage:
    def __init__(
        self,
        update_interval_secs=10,
        update_interval_mb=100,
        remote_file: RemoteFile = None,
    ):
        self._remote_file = remote_file
        self._update_interval_secs = update_interval_secs
        self._update_interval_bytes = update_interval_mb * 1024 * 1024

        self._seen_so_far = 0
        self._lock = Lock()
        self._last_update_time = time.time()
        self._last_update_size_bytes = 0

    def _reset(self):
        self._seen_so_far = 0
        self._lock = Lock()
        self._last_update_time = time.time()
        self._last_update_size_bytes = 0

    def set_file(self, remote_file: RemoteFile):
        self._remote_file = remote_file
        self._reset()

    def __call__(self, bytes_amount):
        if self._remote_file is None:
            raise Exception("remote_file attribute is None.")
        with self._lock:
            self._seen_so_far += bytes_amount
            current_time = time.time()
            if (
                current_time - self._last_update_time >= self._update_interval_secs
                or self._seen_so_far >= (self._last_update_size_bytes + self._update_interval_bytes)
            ):
                self._print_progress()
                self._last_update_time = current_time
                self._last_update_size_bytes = self._seen_so_far

    def _print_progress(self):
        seen_so_far_mb = self._seen_so_far / (1024**2)
        percentage = (seen_so_far_mb / self._remote_file.size_mb) * 100
        # logger.info( #TODO: fix logging
        print(
            f"{self._remote_file.remote_path}  {percentage:.2f}% ({seen_so_far_mb:.1f}/{self._remote_file.size_mb:.1f} MB)\n",
            end="",
            flush=True,
        )

    @staticmethod
    def done_downloading_file(remote_file: RemoteFile):
        logger.info(
            f"Done downloading file: Total Time: {remote_file.download_time:.1f} sec, rate: {remote_file.download_rate_mb_sec:.1f} MB/sec"
        )
