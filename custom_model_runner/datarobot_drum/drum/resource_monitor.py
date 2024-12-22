"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import psutil

from datarobot_drum.drum.enum import ArgumentsOptions
import collections

MemoryInfo = collections.namedtuple(
    "MemoryInfo",
    "total avail free drum_rss container_limit container_max_used container_used",
)


class ResourceMonitor:
    TOTAL_MEMORY_LABEL = "Total"
    AVAILABLE_MEMORY_LABEL = "Available"
    FREE_MEMORY_LABEL = "Free"

    def __init__(self, monitor_current_process=False):
        """"""
        self._is_drum_process = monitor_current_process
        self._current_proc = psutil.Process()
        self._drum_proc = None
        if self._is_drum_process:
            self._drum_proc = self._current_proc

        # save DRUM child processes, because consequent cpu_percent() calls has to be made on the same object.
        self._children_procs = {}

    @staticmethod
    def _run_inside_docker():
        """
        Returns True if running inside a docker container
        """
        if os.path.exists("/.dockerenv"):
            return True
        else:
            return False

    def _collect_memory_info_in_docker(self):
        """
        In the case we are running inside a docker container then memory collection is
        simpler. We look directly on the files inside the /sys/fs/cgroup/memory directory
        and collect the usage/total for all the processes inside the container.
        Returns
        -------
        A dictionary with: {total_mb, usage_mb, max_usage_mb}
        """

        mem_sysfs_path = "/sys/fs/cgroup/memory/"
        total_bytes = int(open(os.path.join(mem_sysfs_path, "memory.limit_in_bytes")).read())
        total_mb = ByteConv.from_bytes(total_bytes).mbytes

        usage_bytes = int(open(os.path.join(mem_sysfs_path, "memory.usage_in_bytes")).read())
        usage_mb = ByteConv.from_bytes(usage_bytes).mbytes

        max_usage_bytes = int(
            open(os.path.join(mem_sysfs_path, "memory.max_usage_in_bytes")).read()
        )
        max_usage_mb = ByteConv.from_bytes(max_usage_bytes).mbytes

        return {"total_mb": total_mb, "usage_mb": usage_mb, "max_usage_mb": max_usage_mb}

    def collect_drum_info(self):
        def get_proc_data(p):
            return {
                "pid": p.pid,
                "cmdline": p.cmdline(),
                "mem": ByteConv.from_bytes(p.memory_info().rss).mbytes,
                "cpu_percent": p.cpu_percent(),
            }

        drum_info = None

        # case with Flask server, there is only one process - drum
        if self._drum_proc is None:
            if self._is_drum_process:
                self._drum_proc = self._current_proc
            else:
                parents = self._current_proc.parents()
                for p in parents:
                    if p.name() == ArgumentsOptions.MAIN_COMMAND:
                        self._drum_proc = p
                        break

        if self._drum_proc:
            drum_info = list()
            drum_info.append(get_proc_data(self._drum_proc))
            if not self._is_drum_process:
                new_children = set()
                for child in self._drum_proc.children(recursive=True):
                    new_children.add(child.pid)
                    if child.pid not in self._children_procs.keys():
                        self._children_procs[child.pid] = child

                # difference is pids in _children_procs and not in new_children
                # remove such processes
                for child_pid in set(self._children_procs.keys()).difference(new_children):
                    self._children_procs.pop(child_pid)

                for child in self._children_procs.values():
                    drum_info.append(get_proc_data(child))

        return drum_info

    def collect_resources_info(self):
        drum_info = self.collect_drum_info()
        virtual_mem = psutil.virtual_memory()
        total_physical_mem_mb = ByteConv.from_bytes(virtual_mem.total).mbytes

        if self._run_inside_docker():
            # RAPTOR-10675: this doesn't work for containers that use cgroups v2, as memory files layout have changed
            container_mem_info = self._collect_memory_info_in_docker()
            container_limit_mb = container_mem_info["total_mb"]
            container_max_usage_mb = container_mem_info["max_usage_mb"]
            container_usage_mb = container_mem_info["usage_mb"]
            available_mem_mb = container_limit_mb - container_mem_info["usage_mb"]
            free_mem_mb = available_mem_mb
        else:
            available_mem_mb = ByteConv.from_bytes(virtual_mem.available).mbytes
            free_mem_mb = ByteConv.from_bytes(virtual_mem.free).mbytes
            container_limit_mb = None
            container_max_usage_mb = None
            container_usage_mb = None

        mem_info = MemoryInfo(
            total=total_physical_mem_mb,
            avail=available_mem_mb,
            free=free_mem_mb,
            drum_rss=sum(info["mem"] for info in drum_info) if self._drum_proc else None,
            container_limit=container_limit_mb,
            container_max_used=container_max_usage_mb,
            container_used=container_usage_mb,
        )

        return {"mem_info": mem_info._asdict(), "drum_info": drum_info if self._drum_proc else None}


class ByteConv(object):
    KB_UNIT = 1024.0

    def __init__(self, size_bytes):
        self._mem_size_bytes = size_bytes

    @classmethod
    def from_bytes(cls, size_bytes):
        return cls(size_bytes)

    @classmethod
    def from_kbytes(cls, size_kbytes):
        return cls(size_kbytes * cls.KB_UNIT)

    @classmethod
    def from_mbytes(cls, size_mbytes):
        return cls(size_mbytes * cls.KB_UNIT * cls.KB_UNIT)

    @classmethod
    def from_gbytes(cls, size_gbytes):
        return cls(size_gbytes * cls.KB_UNIT * cls.KB_UNIT * cls.KB_UNIT)

    @property
    def bytes(self):
        return self._mem_size_bytes

    @property
    def kbytes(self):
        return self._mem_size_bytes / ByteConv.KB_UNIT

    @property
    def mbytes(self):
        return self._mem_size_bytes / ByteConv.KB_UNIT / ByteConv.KB_UNIT

    @property
    def gbytes(self):
        return self._mem_size_bytes / ByteConv.KB_UNIT / ByteConv.KB_UNIT / ByteConv.KB_UNIT
