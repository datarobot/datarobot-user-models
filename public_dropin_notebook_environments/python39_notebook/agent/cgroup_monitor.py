#  Copyright 2022 DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  DataRobot, Inc. Confidential.
#  This is unpublished proprietary source code of DataRobot, Inc.
#  and its affiliates.
#  The copyright notice above does not evidence any actual or intended
#  publication of such source code.
import os
import re
import time

import psutil  # type: ignore

# Parts of this code have been reused from repo:
# https://github.com/neptune-ai/neptune-client/blob/master/LICENSE

NANO_SECS = 10**9


class SystemWatcher:
    @staticmethod
    def cpu_count() -> int:
        return psutil.cpu_count()

    @staticmethod
    def cpu_percent() -> float:
        return psutil.cpu_percent()

    @staticmethod
    def virtual_memory():
        return psutil.virtual_memory()


class CGroupFileReader:
    def __init__(self) -> None:
        cgroup_memory_dir = self._cgroup_mount_dir(subsystem="memory")
        self._memory_usage_file = os.path.join(cgroup_memory_dir, "memory.usage_in_bytes")
        self._memory_limit_file = os.path.join(cgroup_memory_dir, "memory.limit_in_bytes")

        cgroup_cpu_dir = self._cgroup_mount_dir(subsystem="cpu")
        self._cpu_period_file = os.path.join(cgroup_cpu_dir, "cpu.cfs_period_us")
        self._cpu_quota_file = os.path.join(cgroup_cpu_dir, "cpu.cfs_quota_us")

        cgroup_cpuacct_dir = self._cgroup_mount_dir(subsystem="cpuacct")
        self._cpuacct_usage_file = os.path.join(cgroup_cpuacct_dir, "cpuacct.usage")

    def memory_usage_in_bytes(self) -> int:
        return self._read_file(self._memory_usage_file)

    def memory_limit_in_bytes(self) -> int:
        return self._read_file(self._memory_limit_file)

    def cpu_quota_micros(self) -> int:
        return self._read_file(self._cpu_quota_file)

    def cpu_period_micros(self) -> int:
        return self._read_file(self._cpu_period_file)

    def cpuacct_usage_nanos(self) -> int:
        return self._read_file(self._cpuacct_usage_file)

    def _read_file(self, filename: str) -> int:
        with open(filename) as f:
            return int(f.read())

    def _cgroup_mount_dir(self, subsystem: str) -> str:
        """
        :param subsystem: cgroup subsystem like memory, cpu etc.
        :return: directory where subsystem is mounted
        """
        with open("/proc/mounts", "r") as f:
            for line in f.readlines():
                split_line = re.split(r"\s+", line)
                mount_dir = split_line[1]

                if "cgroup" in mount_dir:
                    dirname = mount_dir.split("/")[-1]
                    subsystems = dirname.split(",")
                    if subsystem in subsystems:
                        return mount_dir

        assert False, f'Mount directory for "{subsystem}" subsystem not found'


class CGroupWatcher:
    def __init__(self, cgroup_file_reader: CGroupFileReader, system_watcher: SystemWatcher) -> None:
        self._cgroup_file_reader = cgroup_file_reader
        self._system_watcher = system_watcher

        self._last_cpu_usage_ts_nanos = 0.0
        self._last_cpu_cum_usage_nanos = 0.0

    def memory_usage_in_bytes(self) -> float:
        return self._cgroup_file_reader.memory_usage_in_bytes()

    def memory_limit_in_bytes(self) -> float:
        cgroup_mem_limit = self._cgroup_file_reader.memory_limit_in_bytes()
        total_virtual_memory = self._system_watcher.virtual_memory().total
        return min(cgroup_mem_limit, total_virtual_memory)

    def memory_usage_percentage(self) -> float:
        return round(self.memory_usage_in_bytes() / self.memory_limit_in_bytes() * 100, 2)

    def cpu_usage_limit_in_cores(self) -> float:
        cpu_quota_micros = self._cgroup_file_reader.cpu_quota_micros()

        if cpu_quota_micros == -1:
            return float(self._system_watcher.cpu_count())
        else:
            cpu_period_micros = self._cgroup_file_reader.cpu_period_micros()
            return float(cpu_quota_micros) / float(cpu_period_micros)

    def cpu_usage_percentage(self) -> float:
        current_timestamp_nanos = time.time() * NANO_SECS
        cpu_cum_usage_nanos = self._cgroup_file_reader.cpuacct_usage_nanos()

        if self._is_first_measurement():
            current_usage = 0.0
        else:
            usage_diff = cpu_cum_usage_nanos - self._last_cpu_cum_usage_nanos
            time_diff = current_timestamp_nanos - self._last_cpu_usage_ts_nanos
            current_usage = (
                float(usage_diff) / float(time_diff) / self.cpu_usage_limit_in_cores() * 100.0
            )

        self._last_cpu_usage_ts_nanos = current_timestamp_nanos
        self._last_cpu_cum_usage_nanos = cpu_cum_usage_nanos

        # In case the cpu usage exceeds the limit, we need to limit it
        return round(self._limit(current_usage, lower_limit=0.0, upper_limit=100.0), 2)

    def _is_first_measurement(self) -> bool:
        return self._last_cpu_usage_ts_nanos is None or self._last_cpu_cum_usage_nanos is None

    @staticmethod
    def _limit(value: float, lower_limit: float, upper_limit: float) -> float:
        return max(lower_limit, min(value, upper_limit))
