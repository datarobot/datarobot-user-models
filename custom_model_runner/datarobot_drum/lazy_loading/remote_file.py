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
# from datarobot_custom_code.utils import calculate_rate


class RemoteFile:
    def __init__(self, remote_path, local_path, repository_id):
        self.remote_path = remote_path
        self.local_path = local_path
        self.repository_id = repository_id
        self.size_bytes = None
        self.download_time = None
        self.download_start_time = None
        self.download_status = None
        self.error_msg = None

    @property
    def size_mb(self):
        if self.size_bytes is None:
            return None
        return self.size_bytes / 1024 / 1024

    @property
    def download_rate_mb_sec(self):
        if self.download_time > 0:
            return (self.size_bytes / self.download_time) / (1024 * 1024)
        else:
            return None

    def __str__(self):
        s = f"<RemoteFile {self.remote_path}, local: {self.local_path}, repo: {self.repository_id}"
        if self.size_bytes is None:
            s += ", size: N/A"
        else:
            s += f", size: {self.size_bytes:.1f} MB"

        if self.download_time is None:
            s += ", download_time: N/A"
        else:
            s += f", download_time: {self.download_time:.1f} seconds"

        if self.error_msg is not None:
            s += f"\nerror: {self.error_msg}"
        return s
