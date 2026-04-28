#  Copyright 2022 DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  DataRobot, Inc. Confidential.
#  This is unpublished proprietary source code of DataRobot, Inc.
#  and its affiliates.
#  The copyright notice above does not evidence any actual or intended
#  publication of such source code.
import abc
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import psutil

NANO_SECS = 10 ** 9

if TYPE_CHECKING:
    from psutil._ntuples import svmem


class CGroupVersionUnsupported(Exception):
    """Raised when neither cgroup v1 nor v2 can be detected or used"""


class CGroupFileReaderProtocol(metaclass=abc.ABCMeta):
    """Protocol defining the interface for cgroup file readers (v1 and v2)"""

    @abc.abstractmethod
    def memory_usage_in_bytes(self) -> int:
        """Return current memory usage in bytes"""
        ...

    @abc.abstractmethod
    def memory_limit_in_bytes(self) -> int:
        """Return memory limit in bytes"""
        ...

    @abc.abstractmethod
    def cpu_micros(self) -> tuple[int, int]:
        """Return CPU (quota, period) in microseconds as a tuple"""
        ...

    @abc.abstractmethod
    def cpuacct_usage_nanos(self) -> int:
        """Return cumulative CPU usage in nanoseconds"""
        ...


class SystemWatcher:
    @staticmethod
    def cpu_count() -> int:
        # Returns None if undetermined.
        count = psutil.cpu_count() or 1
        return count

    @staticmethod
    def cpu_percent() -> float:
        # When *percpu* is True returns a list of floats representing the utilization.
        return cast(float, psutil.cpu_percent(percpu=False))

    @staticmethod
    def virtual_memory() -> "svmem":
        return psutil.virtual_memory()


class CGroupFileReader(CGroupFileReaderProtocol):
    def __init__(self) -> None:
        cgroup_memory_dir = self._cgroup_mount_dir(subsystem="memory")
        cgroup_cpu_dir = self._cgroup_mount_dir(subsystem="cpu")
        cgroup_cpuacct_dir = self._cgroup_mount_dir(subsystem="cpuacct")

        self._memory_usage_file = cgroup_memory_dir / "memory.stat"
        self._memory_limit_file = cgroup_memory_dir / "memory.limit_in_bytes"

        self._cpu_period_file = cgroup_cpu_dir / "cpu.cfs_period_us"
        self._cpu_quota_file = cgroup_cpu_dir / "cpu.cfs_quota_us"

        self._cpuacct_usage_file = cgroup_cpuacct_dir / "cpuacct.usage"

    def memory_usage_in_bytes(self) -> int:
        memory_stat_str = self._memory_usage_file.read_text()
        total_rss_str = next(iter([stat for stat in memory_stat_str.split("\n") if stat.startswith("total_rss")]), "0")
        total_rss = int(total_rss_str.split(" ")[-1])
        return total_rss

    def memory_limit_in_bytes(self) -> int:
        return self._read_metric(self._memory_limit_file)

    def cpu_micros(self) -> tuple[int, int]:
        quota = self._read_metric(self._cpu_quota_file)
        period = self._read_metric(self._cpu_period_file)
        return quota, period

    def cpuacct_usage_nanos(self) -> int:
        return self._read_metric(self._cpuacct_usage_file)

    def _read_metric(self, filename: Path) -> int:
        with open(filename) as f:
            return int(f.read())

    def _cgroup_mount_dir(self, subsystem: str) -> Path:
        """
        :param subsystem: cgroup subsystem like memory, cpu etc.
        :return: directory where subsystem is mounted
        """
        try:
            with open("/proc/mounts") as f:
                for line in f.readlines():
                    split_line = re.split(r"\s+", line)
                    mount_dir = split_line[1]

                    if "cgroup" in mount_dir:
                        dirname = mount_dir.split("/")[-1]
                        subsystems = dirname.split(",")

                        if subsystem in subsystems:
                            return Path(mount_dir)
        except FileNotFoundError:
            ...

        raise CGroupVersionUnsupported


class CGroupV2FileReader(CGroupFileReaderProtocol):
    def __init__(self) -> None:
        self._cgroup_base = Path("/sys/fs/cgroup")

        controllers_file = self._cgroup_base / "cgroup.controllers"
        if not controllers_file.exists():
            raise CGroupVersionUnsupported

        # Verify required controllers (cpu and memory) are enabled
        controllers = controllers_file.read_text().strip().split()
        if "cpu" not in controllers or "memory" not in controllers:
            raise CGroupVersionUnsupported

        self._memory_stat_file = self._cgroup_base / "memory.stat"
        self._memory_max_file = self._cgroup_base / "memory.max"
        self._cpu_max_file = self._cgroup_base / "cpu.max"
        self._cpu_stat_file = self._cgroup_base / "cpu.stat"

        # Verify required metric files exist (root cgroup has controllers but no limit files)
        if not self._memory_stat_file.exists():
            raise CGroupVersionUnsupported
        if not self._memory_max_file.exists():
            raise CGroupVersionUnsupported
        if not self._cpu_max_file.exists():
            raise CGroupVersionUnsupported
        if not self._cpu_stat_file.exists():
            raise CGroupVersionUnsupported

    def memory_usage_in_bytes(self) -> int:
        # Read 'anon' from memory.stat to match v1's total_rss behavior (anonymous memory only)
        memory_stat_str = self._memory_stat_file.read_text()
        for line in memory_stat_str.split("\n"):
            if line.startswith("anon "):
                return int(line.split()[1])
        raise ValueError("anon not found in memory.stat")

    def memory_limit_in_bytes(self) -> int:
        limit_str = self._memory_max_file.read_text().strip()
        if limit_str == "max":
            return 2**63 - 1
        return int(limit_str)

    def cpu_micros(self) -> tuple[int, int]:
        """Read cpu.max file and return (quota_micros, period_micros)"""
        cpu_max_str = self._cpu_max_file.read_text().strip()
        quota_str, period_str = cpu_max_str.split()
        quota = -1 if quota_str == "max" else int(quota_str)
        period = int(period_str)
        return quota, period

    def cpuacct_usage_nanos(self) -> int:
        cpu_stat_str = self._cpu_stat_file.read_text()
        for line in cpu_stat_str.split("\n"):
            if line.startswith("usage_usec"):
                usage_usec = int(line.split()[1])
                return usage_usec * 1000
        raise ValueError("usage_usec not found in cpu.stat")


class BaseWatcher:
    def cpu_usage_percentage(self) -> float:
        raise NotImplementedError

    def memory_usage_percentage(self) -> float:
        raise NotImplementedError


class CGroupWatcher(BaseWatcher):
    def __init__(self, cgroup_file_reader: CGroupFileReaderProtocol, system_watcher: SystemWatcher) -> None:
        self._cgroup_file_reader = cgroup_file_reader
        self._system_watcher = system_watcher

        self._last_cpu_usage_ts_nanos = 0.0
        self._last_cpu_cum_usage_nanos = 0.0

    def memory_usage_in_bytes(self) -> float:
        return self._cgroup_file_reader.memory_usage_in_bytes()

    def memory_limit_in_bytes(self) -> float:
        cgroup_mem_limit = self._cgroup_file_reader.memory_limit_in_bytes()
        total_virtual_memory: int = self._system_watcher.virtual_memory().total
        return min(cgroup_mem_limit, total_virtual_memory)

    def memory_usage_percentage(self) -> float:
        return round(self.memory_usage_in_bytes() / self.memory_limit_in_bytes() * 100, 2)

    def cpu_usage_limit_in_cores(self) -> float:
        cpu_quota_micros, cpu_period_micros = self._cgroup_file_reader.cpu_micros()

        if cpu_quota_micros == -1:
            return float(self._system_watcher.cpu_count())
        else:
            return float(cpu_quota_micros) / float(cpu_period_micros)

    def cpu_usage_percentage(self) -> float:
        current_timestamp_nanos = time.time() * NANO_SECS
        cpu_cum_usage_nanos = self._cgroup_file_reader.cpuacct_usage_nanos()

        if self._is_first_measurement():
            current_usage = 0.0
        else:
            usage_diff = cpu_cum_usage_nanos - self._last_cpu_cum_usage_nanos
            time_diff = current_timestamp_nanos - self._last_cpu_usage_ts_nanos
            current_usage = float(usage_diff) / float(time_diff) / self.cpu_usage_limit_in_cores() * 100.0

        self._last_cpu_usage_ts_nanos = current_timestamp_nanos
        self._last_cpu_cum_usage_nanos = cpu_cum_usage_nanos

        # In case the cpu usage exceeds the limit, we need to limit it
        return round(self._limit(current_usage, lower_limit=0.0, upper_limit=100.0), 2)

    def _is_first_measurement(self) -> bool:
        return self._last_cpu_usage_ts_nanos is None or self._last_cpu_cum_usage_nanos is None

    @staticmethod
    def _limit(value: float, lower_limit: float, upper_limit: float) -> float:
        return max(lower_limit, min(value, upper_limit))


class DummyWatcher(BaseWatcher):
    def __init__(self) -> None:
        self._system_watcher = SystemWatcher()

    def cpu_usage_percentage(self) -> float:
        return self._system_watcher.cpu_percent()

    def memory_usage_percentage(self) -> float:
        usage_percentage: float = self._system_watcher.virtual_memory().percent
        return usage_percentage
